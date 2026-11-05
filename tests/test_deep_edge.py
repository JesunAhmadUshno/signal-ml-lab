import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))

from api import InferenceService, serve  # noqa: E402
from edge import artifact_size_bytes, build_quantized_model, load_test_set  # noqa: E402
from signal_ml import extract_features, make_dataset  # noqa: E402


def clean_signal():
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    return np.sin(t) + 0.3 * np.sin(3 * t)


def defect_signal():
    sig = clean_signal().copy()
    sig[20:25] += 1.2
    return sig


class DeepApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = InferenceService()
        cls.server = serve(cls.service, host="127.0.0.1", port=18766)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:18766"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base + "/predict", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_artifact_loads_through_api_path(self):
        self.assertEqual(self.service.manifest["version"], 1)
        self.assertEqual(self.service.model.W1.shape, (12, 16))
        self.assertEqual(self.service.mu.shape, (12,))

    def test_health_endpoint(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(body["status"], "ok")

    def test_predict_clean_inprocess(self):
        result = self.service.predict_signal(clean_signal())
        self.assertEqual(result["class"], 0)
        self.assertEqual(result["label"], "clean")
        self.assertGreater(result["score"], 0.5)

    def test_predict_defect_inprocess(self):
        result = self.service.predict_signal(defect_signal())
        self.assertEqual(result["class"], 1)
        self.assertEqual(result["label"], "defect")
        self.assertGreater(result["score"], 0.5)

    def test_predict_clean_and_defect_http(self):
        status, body = self._post({"signal": clean_signal().tolist()})
        self.assertEqual(status, 200)
        self.assertEqual(body["class"], 0)
        status, body = self._post(defect_signal().tolist())
        self.assertEqual(status, 200)
        self.assertEqual(body["class"], 1)

    def test_bad_input_gives_400_http(self):
        status, _body = self._post({"signal": [1.0, 2.0]})
        self.assertEqual(status, 400)
        status, _body = self._post({"signal": "not-a-list"})
        self.assertEqual(status, 400)
        req = urllib.request.Request(self.base + "/predict", data=b"{not json",
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                code = resp.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        self.assertEqual(code, 400)

    def test_bad_input_inprocess_raises(self):
        with self.assertRaises(ValueError):
            self.service.predict_signal([1.0, 2.0])
        with self.assertRaises(ValueError):
            self.service.predict_signal(["a"] * 64)


class DeepEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = InferenceService()

    def test_quantization_smaller_and_runs(self):
        q_model, q_mu, q_sd, q_size, float_size = build_quantized_model(self.service)
        self.assertLess(q_size, float_size)
        _signals, labels, feats = load_test_set()
        xn = (feats - q_mu) / q_sd
        pred = q_model.predict(xn)
        acc = float(np.mean(pred == labels))
        self.assertGreaterEqual(acc, 0.90)
        sizes = artifact_size_bytes(self.service.artifact_base, self.service.version)
        self.assertGreater(sizes["total_bytes"], 0)
        self.assertGreater(sizes["weight_files_bytes"], 0)

    def test_feature_parity_with_build_features(self):
        sig = defect_signal()
        feats = extract_features(sig)
        cls, _probs = self.service.predict_features(feats)
        self.assertEqual(int(cls[0]), self.service.predict_signal(sig)["class"])
        X, _y = make_dataset(n_per_class=5, seed=0)
        self.assertTrue(np.allclose(extract_features(X[0]), extract_features(np.array(X[0]))))
        self.assertEqual(extract_features(X[0]).shape, (12,))


if __name__ == "__main__":
    unittest.main()
