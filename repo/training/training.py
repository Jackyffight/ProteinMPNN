import argparse
import os.path

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor    
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    base_folder = args.path_for_outputs

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    metrics_file = base_folder + 'metrics.jsonl'
    eval_results_file = base_folder + 'eval_results.json'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
        with open(metrics_file, 'w') as f:
            pass

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': args.num_loader_workers}
    if args.seed >= 0:
        LOAD_PARAM['generator'] = torch.Generator().manual_seed(args.seed)

   
    if args.debug:
        print(
            "[debug] debug mode overrides num_examples_per_epoch, max_protein_length, and "
            "batch_size to 50/1000/1000; pass --debug False to honor explicitly-set values.",
            flush=True,
        )
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime()),
        "args": vars(args),
        "data": {
            "path_for_training_data": data_path,
            "list": params["LIST"],
            "valid_clusters": params["VAL"],
            "test_clusters": params["TEST"],
            "train_cluster_count": len(train),
            "valid_cluster_count": len(valid),
            "test_cluster_count": len(test),
            "rescut": args.rescut,
            "homology_cutoff": params["HOMO"],
        },
        "model": {
            "hidden_dim": args.hidden_dim,
            "num_encoder_layers": args.num_encoder_layers,
            "num_decoder_layers": args.num_decoder_layers,
            "num_neighbors": args.num_neighbors,
            "dropout": args.dropout,
            "backbone_noise": args.backbone_noise,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    with open(base_folder + 'run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
     
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)


    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_decoder_layers,
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)


    if PATH:
        checkpoint = torch.load(PATH, map_location=device)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        best_validation_loss = checkpoint.get('best_validation_loss', float('inf'))
    else:
        total_step = 0
        epoch = 0
        best_validation_loss = float('inf')

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)


    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    with ProcessPoolExecutor(max_workers=args.prefetch_workers) as executor:
        q = queue.Queue(maxsize=args.prefetch_batches)
        p = queue.Queue(maxsize=args.prefetch_batches)
        for i in range(args.prefetch_batches):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
       
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
           
                    scaler.scale(loss_av_smoothed).backward()

                    if args.gradient_norm > 0.0:
                        # Unscale before clipping so the clip threshold applies to real
                        # gradients, not GradScaler-scaled ones. scaler.step reuses this
                        # unscale (keyed on the optimizer object) and does not repeat it.
                        scaler.unscale_(optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = float(train_sum / train_weights)
            train_accuracy = float(train_acc / train_weights)
            train_perplexity = np.exp(train_loss)
            validation_loss = float(validation_sum / validation_weights)
            validation_accuracy = float(validation_acc / validation_weights)
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')

            epoch_metrics = {
                "epoch": int(e + 1),
                "step": int(total_step),
                "seconds": float(t1 - t0),
                "train_loss": train_loss,
                "train_perplexity": float(train_perplexity),
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_perplexity": float(validation_perplexity),
                "validation_accuracy": validation_accuracy,
                "num_examples_per_epoch": int(args.num_examples_per_epoch),
                "batch_tokens": int(args.batch_size),
                "max_protein_length": int(args.max_protein_length),
            }
            is_best = validation_loss < best_validation_loss

            with open(metrics_file, 'a') as f:
                f.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
            # eval_results.json tracks the BEST epoch (matching model_weights/best.pt
            # when --save_best), not the last, so a consumer reads consistent metrics.
            if is_best or not os.path.exists(eval_results_file):
                with open(eval_results_file, 'w') as f:
                    json.dump(
                        {
                            "epoch": epoch_metrics["epoch"],
                            "step": epoch_metrics["step"],
                            "eval_loss": epoch_metrics["validation_loss"],
                            "eval_perplexity": epoch_metrics["validation_perplexity"],
                            "eval_accuracy": epoch_metrics["validation_accuracy"],
                            "selection": "best_validation_loss",
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )

            checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'
            checkpoint_payload = {
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        'best_validation_loss': min(best_validation_loss, validation_loss),
                        'config': manifest["model"],
                        'metrics': epoch_metrics,
                        }
            torch.save({
                        **checkpoint_payload,
                        }, checkpoint_filename_last)

            if is_best:
                best_validation_loss = validation_loss
                if args.save_best:
                    torch.save({
                                **checkpoint_payload,
                                'best_validation_loss': best_validation_loss,
                                }, base_folder+'model_weights/best.pt')

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        **checkpoint_payload,
                        }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=str2bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
    argparser.add_argument("--seed", type=int, default=42, help="random seed; set to a negative value to leave RNG unseeded")
    argparser.add_argument("--num_loader_workers", type=int, default=4, help="PyTorch DataLoader workers for raw PDB loading")
    argparser.add_argument("--prefetch_workers", type=int, default=12, help="ProcessPool workers for structure prefetch")
    argparser.add_argument("--prefetch_batches", type=int, default=3, help="number of prefetched train/validation structure batches")
    argparser.add_argument("--tf32", type=str2bool, default=True, help="allow TF32 matmul on Ampere and newer GPUs")
    argparser.add_argument("--save_best", type=str2bool, default=True, help="write model_weights/best.pt when validation improves")
 
    args = argparser.parse_args()    
    main(args)   
