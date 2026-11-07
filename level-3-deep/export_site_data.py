"""Export the committed model and the seed-42 test set for the live dashboard.

Writes two JSON files under site/src/generated/:
- model.json: the Build artifact arrays (W1, b1, W2, b2, mu, sd) read
  through the hash-verified load_artifact, plus the manifest hash.
- testset.json: the exact 500 synthetic test signals used by Build and
  Deep (make_dataset n_per_class=250, seed=42), their labels and defect
  parameters, and the Python features, p(defect), predictions and int8
  predictions (edge.py quantization) that the site's build-time check
  compares the JavaScript port against.

Floats are written with Python's shortest round-trip repr, so the
browser sees exactly the float64 values NumPy used. NumPy only, under a
second. Usage: python level-3-deep/export_site_data.py
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from edge import build_quantized_model  # noqa: E402
from rigor import TEST_PER_CLASS, TEST_SEED, committed_model, dataset_with_params, feats  # noqa: E402

OUT = os.path.join(os.path.dirname(_HERE), "site", "src", "generated")


def export(out_dir=OUT):
    model, mu, sd = committed_model()
    with open(os.path.join(os.path.dirname(_HERE), "level-2-build", "artifacts", "model", "1", "manifest.json")) as f:
        manifest = json.load(f)
    signals, labels, params = dataset_with_params(TEST_PER_CLASS, seed=TEST_SEED)
    X = feats(signals)
    p = model.forward((X - mu) / sd)
    q_model, q_mu, q_sd, _q, _f = build_quantized_model(SimpleNamespace(model=model, mu=mu, sd=sd))
    q_pred = q_model.predict((X - q_mu) / q_sd)
    model_json = {
        "source": "level-2-build/artifacts/model/1 via artifact.load_artifact",
        "weights_sha256": manifest["weights_sha256"],
        "W1": model.W1.tolist(), "b1": model.b1.tolist(), "W2": model.W2.tolist(), "b2": model.b2.tolist(),
        "mu": mu.tolist(), "sd": sd.tolist(),
    }
    test_json = {
        "source": "make_dataset(n_per_class=250, seed=42), synthetic only; written by export_site_data.py",
        "seed": TEST_SEED,
        "labels": labels.tolist(),
        "params": params,
        "signals": signals.tolist(),
        "py_features": X.tolist(),
        "py_p_defect": p[:, 1].tolist(),
        "py_pred": np.argmax(p, axis=1).tolist(),
        "py_pred_int8": q_pred.tolist(),
    }
    os.makedirs(out_dir, exist_ok=True)
    for name, obj in (("model.json", model_json), ("testset.json", test_json)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, separators=(",", ":"))
            f.write("\n")
    return model_json, test_json


if __name__ == "__main__":
    _m, t = export()
    print("exported", len(t["labels"]), "signals; python correct:",
          int(np.sum(np.array(t["py_pred"]) == np.array(t["labels"]))),
          "int8 correct:", int(np.sum(np.array(t["py_pred_int8"]) == np.array(t["labels"]))))
