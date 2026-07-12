import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HAS_BUILDER_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("Bio", "numpy", "torch")
)


@unittest.skipUnless(HAS_BUILDER_DEPS, "mmCIF builder dependencies not installed")
class MmcifDatasetV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "repo/training"))
        global builder, crop_builder, np, torch
        import build_pdb_mmcif_dataset as builder
        import build_pdb_oversized_crop_tar_dataset as crop_builder
        import numpy as np
        import torch

    @classmethod
    def tearDownClass(cls):
        sys.path.pop(0)

    def test_polymer_sequence_positions_are_preserved_when_coordinates_are_missing(self):
        cif = {
            "_entity_poly.entity_id": ["1"],
            "_entity_poly.type": ["polypeptide(L)"],
            "_entity_poly_seq.entity_id": ["1", "1", "1", "1"],
            "_entity_poly_seq.num": ["1", "2", "3", "4"],
            "_entity_poly_seq.mon_id": ["ALA", "GLY", "SER", "VAL"],
            "_pdbx_poly_seq_scheme.asym_id": ["A", "A", "A", "A"],
            "_pdbx_poly_seq_scheme.entity_id": ["1", "1", "1", "1"],
            "_pdbx_poly_seq_scheme.seq_id": ["1", "2", "3", "4"],
            "_pdbx_poly_seq_scheme.mon_id": ["ALA", "GLY", "SER", "VAL"],
        }
        atom_names = ["N", "CA", "C", "O"] * 2
        cif.update(
            {
                "_atom_site.label_atom_id": atom_names,
                "_atom_site.label_comp_id": ["ALA"] * 4 + ["SER"] * 4,
                "_atom_site.label_asym_id": ["A"] * 8,
                "_atom_site.label_entity_id": ["1"] * 8,
                "_atom_site.label_seq_id": ["1"] * 4 + ["3"] * 4,
                "_atom_site.label_alt_id": ["."] * 8,
                "_atom_site.group_PDB": ["ATOM"] * 8,
                "_atom_site.Cartn_x": [str(i) for i in range(8)],
                "_atom_site.Cartn_y": ["0"] * 8,
                "_atom_site.Cartn_z": ["0"] * 8,
                "_atom_site.occupancy": ["1"] * 8,
                "_atom_site.B_iso_or_equiv": ["10"] * 8,
                "_atom_site.pdbx_PDB_model_num": ["1"] * 8,
            }
        )

        chains = builder.extract_polymer_chains(
            cif,
            {
                "min_chain_length": 1,
                "max_chain_length": 100,
                "min_resolved_residues": 1,
                "min_backbone_coverage": 0.0,
            },
        )

        chain = chains["A"]
        self.assertEqual(chain["seq"], "AGSV")
        self.assertEqual(tuple(chain["xyz"].shape), (4, 14, 3))
        self.assertEqual(chain["resolved_residue_count"], 2)
        self.assertTrue(np.isnan(chain["xyz"][1].numpy()).all())
        self.assertFalse(chain["mask"][1].any().item())

    def test_canonical_discovery_keeps_one_lowest_assembly_per_pdb(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in (
                "5naf-assembly4.cif.gz",
                "5naf-assembly2.cif.gz",
                "5naf-assembly1.cif.gz",
                "1abc-assembly2.cif.gz",
            ):
                (root / name).touch()

            files = builder.discover_files(root, "all", 0, assembly_policy="first")

        self.assertEqual(
            [path.name for path in files],
            ["1abc-assembly2.cif.gz", "5naf-assembly1.cif.gz"],
        )

    def test_target_selection_prefers_resolved_backbone_then_source_id(self):
        chains = {
            "B": {"resolved_residue_count": 8, "backbone_coverage": 0.8},
            "A": {"resolved_residue_count": 8, "backbone_coverage": 0.8},
            "C": {"resolved_residue_count": 7, "backbone_coverage": 1.0},
        }
        self.assertEqual(builder.select_target_chain(chains), "A")

    def test_context_length_counts_full_polymer_sequences(self):
        chains = {"A": {"seq": "A" * 1200}, "B": {"seq": "G" * 900}}
        self.assertEqual(builder.total_context_length(chains), 2100)

    def make_chain(self, length, ca_x, resolved_count=None):
        xyz = torch.full((length, 14, 3), float("nan"))
        mask = torch.zeros((length, 14), dtype=torch.bool)
        for atom_index in range(4):
            xyz[:, atom_index, 0] = torch.as_tensor(ca_x, dtype=torch.float32)
            xyz[:, atom_index, 1:] = 0.0
            mask[:, atom_index] = True
        resolved_count = length if resolved_count is None else resolved_count
        return {
            "seq": "A" * length,
            "xyz": xyz,
            "mask": mask,
            "bfac": torch.zeros((length, 14)),
            "occ": torch.ones((length, 14)),
            "entity_id": "1",
            "source_chain_id": "unused",
            "resolved_residue_count": resolved_count,
            "backbone_coverage": resolved_count / float(length),
        }

    def test_spatial_crop_keeps_target_and_contiguous_nearest_context_window(self):
        target = self.make_chain(100, [0.0] * 100)
        near = self.make_chain(80, [abs(index - 40) + 1.0 for index in range(80)])
        far = self.make_chain(70, [100.0] * 70)
        near["entity_id"] = "2"
        far["entity_id"] = "3"
        chains = {"A": target, "B": near, "C": far}

        result = builder.crop_spatial_context(
            chains,
            max_context_length=150,
            max_chains=62,
            min_context_crop_length=30,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["chains"]), ["A", "B"])
        self.assertEqual(builder.total_context_length(result["chains"]), 150)
        context = result["chains"]["B"]
        self.assertEqual(context["crop"]["kind"], "context_window")
        self.assertEqual(context["crop"]["source_residue_start"], 15)
        self.assertEqual(context["crop"]["source_residue_end"], 65)
        self.assertEqual(context["crop"]["nearest_source_residue"], 40)
        self.assertEqual(context["crop"]["source_sequence_length"], 80)
        self.assertEqual(context["crop"]["source_resolved_residue_count"], 80)
        self.assertEqual(len(context["seq"]), 50)
        self.assertEqual(context["xyz"].untyped_storage().nbytes(), context["xyz"].numel() * 4)
        self.assertFalse(context["xyz"].untyped_storage().data_ptr() == near["xyz"].untyped_storage().data_ptr())

    def test_spatial_crop_rejects_target_longer_than_budget(self):
        chains = {"A": self.make_chain(101, [0.0] * 101)}

        result = builder.crop_spatial_context(
            chains,
            max_context_length=100,
            max_chains=62,
            min_context_crop_length=30,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "target_too_long_for_complete_crop")
        self.assertEqual(result["target_length"], 101)

    def test_spatial_crop_skips_a_sparse_local_window(self):
        target = self.make_chain(70, [0.0] * 70)
        sparse = self.make_chain(100, [100.0] * 100)
        sparse["xyz"][:] = float("nan")
        sparse["mask"][:] = False
        for residue in list(range(30)) + [80]:
            sparse["xyz"][residue, :4, 0] = 100.0 if residue < 30 else 1.0
            sparse["xyz"][residue, :4, 1:] = 0.0
            sparse["mask"][residue, :4] = True
        sparse["resolved_residue_count"] = 31
        sparse["backbone_coverage"] = 0.31
        dense = self.make_chain(30, [10.0] * 30)
        sparse["entity_id"] = "2"
        dense["entity_id"] = "3"

        result = builder.crop_spatial_context(
            {"A": target, "B": sparse, "C": dense},
            max_context_length=100,
            max_chains=62,
            min_context_crop_length=30,
            min_resolved_residues=30,
            min_backbone_coverage=0.5,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(list(result["chains"]), ["A", "C"])
        self.assertEqual(builder.total_context_length(result["chains"]), 100)

    def test_oversized_split_inheritance_uses_reference_clusters_and_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir)
            (reference / "list.csv").write_text(
                "CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE\n"
                "1abca1_A,2025-01-01,2.0,h1,10,AAAA\n"
                "2abca1_A,2025-01-01,2.0,h2,20,BBBB\n"
                "3abca1_A,2025-01-01,2.0,h3,30,CCCC\n",
                encoding="utf-8",
            )
            (reference / "valid_clusters.txt").write_text("20\n", encoding="utf-8")
            (reference / "test_clusters.txt").write_text("30\n", encoding="utf-8")
            rows = [
                {"CHAINID": "4abca1_A", "CLUSTER": "99", "SEQUENCE": "BBBB"},
                {"CHAINID": "5abca1_A", "CLUSTER": "40", "SEQUENCE": "DDDD"},
                {"CHAINID": "6abca1_A", "CLUSTER": "30", "SEQUENCE": "EEEE"},
            ]

            stats, quarantined = crop_builder.inherit_reference_splits(rows, reference)

        self.assertEqual([row["CLUSTER"] for row in rows], ["20", "40", "30"])
        self.assertEqual(quarantined, [])
        self.assertEqual(stats["stage_valid_cluster_ids"], [20])
        self.assertEqual(stats["stage_test_cluster_ids"], [30])
        self.assertEqual(stats["stage_train_rows"], 1)
        self.assertEqual(stats["stage_valid_rows"], 1)
        self.assertEqual(stats["stage_test_rows"], 1)

    def test_oversized_split_inheritance_quarantines_a_conflicting_component(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reference = Path(temp_dir)
            (reference / "list.csv").write_text(
                "CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE\n"
                "1abca1_A,2025-01-01,2.0,h1,10,AAAA\n"
                "2abca1_A,2025-01-01,2.0,h2,20,BBBB\n",
                encoding="utf-8",
            )
            (reference / "valid_clusters.txt").write_text("20\n", encoding="utf-8")
            (reference / "test_clusters.txt").write_text("", encoding="utf-8")
            rows = [
                {"CHAINID": "3abca1_A", "CLUSTER": "10", "SEQUENCE": "BBBB"},
                {"CHAINID": "4abca1_A", "CLUSTER": "10", "SEQUENCE": "CCCC"},
                {"CHAINID": "5abca1_A", "CLUSTER": "30", "SEQUENCE": "DDDD"},
            ]

            stats, quarantined = crop_builder.inherit_reference_splits(rows, reference)

        self.assertEqual([row["CHAINID"] for row in rows], ["5abca1_A"])
        self.assertEqual(stats["split_conflict_components_quarantined"], 1)
        self.assertEqual(stats["split_conflict_rows_quarantined"], 2)
        self.assertEqual(
            {row["chain_id"] for row in quarantined},
            {"3abca1_A", "4abca1_A"},
        )
        self.assertEqual(
            {tuple(row["reference_splits"]) for row in quarantined},
            {("train", "valid")},
        )

    def test_oversized_index_rewrite_excludes_quarantined_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chain_index = root / "index.jsonl"
            record_index = root / "records.jsonl"
            chain_index.write_text(
                json.dumps({"chain_id": "1abca1_A", "cluster": 10}) + "\n"
                + json.dumps({"chain_id": "2abca1_A", "cluster": 20}) + "\n",
                encoding="utf-8",
            )
            record_index.write_text(
                json.dumps({"entry_id": "1abca1", "chains": ["1abca1_A"], "clusters": [10]}) + "\n"
                + json.dumps({"entry_id": "2abca1", "chains": ["2abca1_A"], "clusters": [20]}) + "\n",
                encoding="utf-8",
            )

            crop_builder.filter_and_rewrite_indexes(
                chain_index,
                record_index,
                [{"CHAINID": "2abca1_A", "CLUSTER": "30"}],
            )

            chain_rows = [json.loads(line) for line in chain_index.read_text().splitlines()]
            record_rows = [json.loads(line) for line in record_index.read_text().splitlines()]

        self.assertEqual(chain_rows, [{"chain_id": "2abca1_A", "cluster": 30}])
        self.assertEqual(
            record_rows,
            [{"chains": ["2abca1_A"], "clusters": [30], "entry_id": "2abca1"}],
        )

    def test_exact_sequence_cluster_conflicts_are_unioned(self):
        rows = [
            {"CHAINID": "1abca1_A", "CLUSTER": "11", "SEQUENCE": "AAAA"},
            {"CHAINID": "2abca1_A", "CLUSTER": "22", "SEQUENCE": "AAAA"},
            {"CHAINID": "3abca1_A", "CLUSTER": "22", "SEQUENCE": "AAAT"},
        ]

        stats = builder.reconcile_exact_sequence_clusters(rows)

        self.assertEqual({row["CLUSTER"] for row in rows}, {"11"})
        self.assertEqual(stats["exact_sequence_conflicts_before"], 1)
        self.assertEqual(stats["clusters_merged"], 1)

    def test_aligned_identity_handles_an_internal_insertion(self):
        self.assertAlmostEqual(builder.sequence_identity("ACDE", "ACXDE"), 0.8)

    def test_entry_metadata_date_filter_runs_before_mmcif_decompression(self):
        previous_metadata = builder.ENTRY_METADATA
        builder.ENTRY_METADATA = {
            "1ABC": {
                "date": "2020-01-01",
                "method": "X-RAY DIFFRACTION",
                "resolution": 2.0,
            }
        }
        config = {
            "min_date": "2021-08-03",
            "max_date": "2026-07-08",
            "method_allow": {"X-RAY DIFFRACTION"},
            "max_resolution": 3.5,
        }
        try:
            with mock.patch.object(
                builder,
                "read_mmcif",
                side_effect=AssertionError("old entries must not be decompressed"),
            ):
                result = builder.parse_one("/tmp/1abc-assembly1.cif.gz", config)
        finally:
            builder.ENTRY_METADATA = previous_metadata

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "date")

    def test_payload_keeps_target_ids_but_not_provisional_cluster_rows(self):
        monomers = ["ALA", "GLY", "SER", "VAL"]
        cif = {
            "_entity_poly.entity_id": ["1"],
            "_entity_poly.type": ["polypeptide(L)"],
            "_entity_poly_seq.entity_id": ["1"] * 4,
            "_entity_poly_seq.num": [str(i) for i in range(1, 5)],
            "_entity_poly_seq.mon_id": monomers,
            "_pdbx_poly_seq_scheme.asym_id": ["A"] * 4,
            "_pdbx_poly_seq_scheme.entity_id": ["1"] * 4,
            "_pdbx_poly_seq_scheme.seq_id": [str(i) for i in range(1, 5)],
            "_pdbx_poly_seq_scheme.mon_id": monomers,
            "_atom_site.label_atom_id": ["N", "CA", "C", "O"] * 4,
            "_atom_site.label_comp_id": [monomer for monomer in monomers for _ in range(4)],
            "_atom_site.label_asym_id": ["A"] * 16,
            "_atom_site.label_entity_id": ["1"] * 16,
            "_atom_site.label_seq_id": [str(i) for i in range(1, 5) for _ in range(4)],
            "_atom_site.label_alt_id": ["."] * 16,
            "_atom_site.group_PDB": ["ATOM"] * 16,
            "_atom_site.Cartn_x": [str(i) for i in range(16)],
            "_atom_site.Cartn_y": ["0"] * 16,
            "_atom_site.Cartn_z": ["0"] * 16,
            "_atom_site.occupancy": ["1"] * 16,
            "_atom_site.B_iso_or_equiv": ["10"] * 16,
            "_atom_site.pdbx_PDB_model_num": ["1"] * 16,
        }
        previous_metadata = builder.ENTRY_METADATA
        previous_clusters = builder.CLUSTER_MAP
        builder.ENTRY_METADATA = {
            "1ABC": {
                "date": "2025-01-01",
                "method": "X-RAY DIFFRACTION",
                "resolution": 2.0,
            }
        }
        builder.CLUSTER_MAP = {("1ABC", "1"): 42}
        config = {
            "out_dir": "/tmp/unused",
            "min_date": "2021-08-03",
            "max_date": "2026-07-08",
            "method_allow": {"X-RAY DIFFRACTION"},
            "max_resolution": 3.5,
            "min_chain_length": 1,
            "max_chain_length": 100,
            "max_context_length": 100,
            "min_resolved_residues": 1,
            "min_backbone_coverage": 0.0,
            "max_chains": 62,
            "write_pt": False,
            "return_payload": True,
        }
        try:
            with mock.patch.object(builder, "read_mmcif", return_value=cif):
                result = builder.parse_one("/tmp/1abc-assembly1.cif.gz", config)
        finally:
            builder.ENTRY_METADATA = previous_metadata
            builder.CLUSTER_MAP = previous_clusters

        payload = torch.load(io.BytesIO(result["payload"]), weights_only=True)
        self.assertEqual(payload["target_chain_ids"], ["1abca1_A"])
        self.assertNotIn("rows", payload)


if __name__ == "__main__":
    unittest.main()
