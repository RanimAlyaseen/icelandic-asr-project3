"""Checks for the separate classical forced-alignment experiment."""
import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from mfcc_gmm_hmm.iterative_alignment import forced_align, refit
from mfcc_gmm_hmm.compare_alignment import paired_dev_uncertainty


class AcousticPrototype:
    def __init__(self, value):
        self.value = value

    def score_samples(self, features):
        return -20.0 * (features[:, 0] - self.value) ** 2


class ForcedAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.model = {"tokens": ["a", "b"], "models": [AcousticPrototype(0), AcousticPrototype(1)],
                      "self_loop": 0.75}

    def test_unequal_phone_durations_are_not_uniform(self):
        features = np.array([[0.0]] * 5 + [[1.0]] * 2, dtype=np.float32)
        self.assertEqual(forced_align(features, "ab", self.model).tolist(), [0, 0, 0, 0, 0, 1, 1])

    def test_repeated_letters_occupy_distinct_positions(self):
        features = np.array([[0.0]] * 4 + [[1.0]] * 2, dtype=np.float32)
        path = forced_align(features, "aab", self.model)
        self.assertEqual(set(path), {0, 1, 2})
        self.assertTrue(np.all(np.diff(path) >= 0))

    def test_rejects_unalignable_transcript(self):
        with self.assertRaises(ValueError):
            forced_align(np.zeros((2, 1), dtype=np.float32), "aba", self.model)

    def test_rejects_dev_audio_in_training(self):
        with self.assertRaisesRegex(ValueError, "train rows only"):
            refit([{"split": "dev", "id": "x"}], self.model, 1)

    def test_paired_comparison_rejects_mismatched_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{index}.csv" for index in range(2)]
            for path, ident in zip(paths, ("clip-a", "clip-b")):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("id", "reference", "hypothesis"))
                    writer.writeheader()
                    writer.writerow({"id": ident, "reference": "a b", "hypothesis": "a"})
            with self.assertRaisesRegex(ValueError, "paired"):
                paired_dev_uncertainty(*paths)


if __name__ == "__main__":
    unittest.main()
