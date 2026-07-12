import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo/training"))

from evaluation_utils import flatten_split_records  # noqa: E402


class EvaluationUtilsTest(unittest.TestCase):
    def test_flatten_split_records_is_exhaustive_and_deterministic(self):
        clusters = {
            20: [["2bbb_B", "hash-b"], ["2aaa_A", "hash-a"]],
            10: [["1ccc_C", "hash-c"]],
        }

        records = flatten_split_records(clusters)

        self.assertEqual(
            records,
            [
                ["1ccc_C", "hash-c"],
                ["2aaa_A", "hash-a"],
                ["2bbb_B", "hash-b"],
            ],
        )
        self.assertEqual(len(records), sum(map(len, clusters.values())))


if __name__ == "__main__":
    unittest.main()
