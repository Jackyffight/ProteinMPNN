import importlib.util
import io
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
        global builder, np, torch
        import build_pdb_mmcif_dataset as builder
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
