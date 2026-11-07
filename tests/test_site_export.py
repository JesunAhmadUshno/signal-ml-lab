import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))

from export_site_data import export  # noqa: E402

ART = os.path.join(ROOT, "level-2-build", "artifacts", "model", "1")
DATA = os.path.join(ROOT, "site", "src", "generated")


class SiteExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.model, cls.test = export(cls.tmp)

    def test_weights_are_the_committed_arrays(self):
        for name in ("W1", "b1", "W2", "b2", "mu", "sd"):
            np.testing.assert_array_equal(np.array(self.model[name]), np.load(os.path.join(ART, name + ".npy")))

    def test_python_predictions_reproduce_recorded_accuracy(self):
        with open(os.path.join(ROOT, "level-3-deep", "results_deep.json")) as f:
            deep = json.load(f)["edge_study"]
        y = np.array(self.test["labels"])
        self.assertEqual(int(np.sum(np.array(self.test["py_pred"]) == y)), deep["float_artifact"]["correct"])
        self.assertEqual(int(np.sum(np.array(self.test["py_pred_int8"]) == y)),
                         deep["int8_quantized_numpy_study"]["correct"])

    def test_committed_site_data_is_up_to_date(self):
        for name in ("model.json", "testset.json"):
            with open(os.path.join(self.tmp, name)) as a, open(os.path.join(DATA, name)) as b:
                self.assertEqual(json.load(a), json.load(b), name + " is stale: rerun export_site_data.py")


if __name__ == "__main__":
    unittest.main()
