import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))

from rigor import dataset_with_params, wilson  # noqa: E402
from signal_ml import make_dataset  # noqa: E402


class RigorTests(unittest.TestCase):
    def test_default_generator_matches_make_dataset_draw_for_draw(self):
        for n, seed in [(20, 0), (250, 42)]:
            a, ay = make_dataset(n_per_class=n, seed=seed)
            b, by, params = dataset_with_params(n, seed=seed)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(ay, by)
            self.assertEqual([p is not None for p in params], list(by == 1))

    def test_defect_params_describe_the_bump(self):
        sig, y, params = dataset_with_params(5, seed=3, noise=0.0)
        clean = dataset_with_params(5, seed=3, noise=0.0, amp=(0.0, 0.0))[0]
        for s, c, p in zip(sig, clean, params):
            if p:
                diff = s - c
                self.assertAlmostEqual(diff[p["start"]], p["amp"])
                self.assertEqual(int(np.sum(np.abs(diff) > 1e-12)), p["width"])

    def test_wilson_interval_for_492_of_500(self):
        self.assertEqual(wilson(492, 500), [0.9687, 0.9919])

    def test_committed_rigor_results_agree_with_build(self):
        with open(os.path.join(ROOT, "level-3-deep", "results_rigor.json")) as f:
            rigor = json.load(f)
        with open(os.path.join(ROOT, "level-2-build", "results_build.json")) as f:
            build = json.load(f)
        head = rigor["headline_committed_model"]
        self.assertEqual(head["accuracy"], build["training"]["test_accuracy"])
        self.assertEqual(rigor["error_analysis"]["missed_defect"],
                         250 - build["training"]["per_class_test"]["1"]["correct"])


if __name__ == "__main__":
    unittest.main()
