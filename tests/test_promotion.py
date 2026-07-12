import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo/training"))

from promote_checkpoint import promote_checkpoint  # noqa: E402


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


class PromotionTest(unittest.TestCase):
    def make_run(self, root, test_nll_delta=-0.01):
        run_dir = root / "run"
        checkpoint = run_dir / "model_weights/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint_bytes = b"validated checkpoint fixture"
        checkpoint.write_bytes(checkpoint_bytes)
        checkpoint_sha256 = sha256_bytes(checkpoint_bytes)
        checkpoint_record = {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "metadata": {"epoch": 1, "step": 494},
        }

        valid_summary = {
            "schema": "proteinmpnn.stage1_fixed_valid_summary.v1",
            "selection_metric": "valid_nll",
            "official": {
                "records": 426,
                "nll": 1.6,
                "perplexity": 4.95,
                "accuracy": 0.50,
            },
            "best_candidate": {
                "label": "best",
                "records": 426,
                "checkpoint": checkpoint_record,
                "nll": 1.59,
                "perplexity": 4.90,
                "accuracy": 0.51,
            },
            "best_candidate_delta": {
                "nll": -0.01,
                "perplexity": -0.05,
                "accuracy": 0.01,
            },
            "ranked": [],
        }
        test_summary = {
            "schema": "proteinmpnn.selected_test_summary.v1",
            "official": {"nll": 1.61, "perplexity": 5.00, "accuracy": 0.49},
            "selected_checkpoint": checkpoint_record,
            "selected": {
                "nll": 1.61 + test_nll_delta,
                "perplexity": 4.97,
                "accuracy": 0.50,
            },
            "delta": {
                "nll": test_nll_delta,
                "perplexity": -0.03,
                "accuracy": 0.01,
            },
            "records": 461,
        }
        valid_path = run_dir / "evaluations/fixed-valid-records/summary.json"
        test_path = run_dir / "evaluations/selected-test-records/summary.json"
        valid_path.parent.mkdir(parents=True)
        test_path.parent.mkdir(parents=True)
        valid_path.write_text(json.dumps(valid_summary), encoding="utf-8")
        test_path.write_text(json.dumps(test_summary), encoding="utf-8")
        return run_dir, checkpoint_sha256

    def test_promotes_checkpoint_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, expected_sha256 = self.make_run(root)
            destination = root / "promoted"

            first = promote_checkpoint(run_dir, destination, "fixture-model")
            second = promote_checkpoint(run_dir, destination, "fixture-model")

            self.assertEqual(first, second)
            self.assertEqual(first["checkpoint"]["sha256"], expected_sha256)
            self.assertEqual(first["checkpoint"]["metadata"]["epoch"], 1)
            self.assertEqual((destination / "model.pt").read_bytes(), b"validated checkpoint fixture")
            self.assertTrue((destination / "fixed-valid-summary.json").is_file())
            self.assertTrue((destination / "selected-test-summary.json").is_file())
            self.assertTrue((destination / "promotion.json").is_file())

    def test_rejects_checkpoint_that_did_not_improve_test_nll(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir, _ = self.make_run(root, test_nll_delta=0.01)

            with self.assertRaisesRegex(ValueError, "test NLL did not improve"):
                promote_checkpoint(run_dir, root / "promoted", "fixture-model")


if __name__ == "__main__":
    unittest.main()
