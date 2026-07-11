import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class CheckpointUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "repo/training"))
        global checkpoint_utils, torch
        import checkpoint_utils
        import torch

    @classmethod
    def tearDownClass(cls):
        sys.path.pop(0)

    def test_weights_only_checkpoint_initializes_a_model(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "official_style.pt"
            torch.save(
                {
                    "num_edges": 48,
                    "noise_level": 0.2,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            checkpoint = checkpoint_utils.load_checkpoint(checkpoint_path)
            target = torch.nn.Linear(3, 2)
            checkpoint_utils.validate_num_edges(checkpoint, 48)
            checkpoint_utils.load_model_weights(target, checkpoint, checkpoint_path)

            for expected, actual in zip(model.parameters(), target.parameters()):
                self.assertTrue(torch.equal(expected, actual))

    def test_resume_rejects_a_weights_only_checkpoint(self):
        checkpoint = {"model_state_dict": {}}
        with self.assertRaisesRegex(ValueError, "use --init_checkpoint"):
            checkpoint_utils.require_resume_state(checkpoint, "official.pt")

    def test_neighbor_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_edges=48"):
            checkpoint_utils.validate_num_edges({"num_edges": 48}, 32)
