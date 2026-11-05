"""Level 2 Build: versioned model artifact discipline, NumPy format.

This mirrors the SavedModel ideas in SPEC Build (signatures,
versioning, load checks) in a labelled NumPy artifact format. It is
NOT SavedModel and is never described as SavedModel: TensorFlow is
not available in this environment (verified 2026-11-05).

Layout:
    <base_dir>/model/<version>/manifest.json
    <base_dir>/model/<version>/W1.npy, b1.npy, W2.npy, b2.npy, mu.npy, sd.npy

The manifest records the format name, the version, the input
signature (feature dim 12), the output signature (2 classes), the
training config, and a sha256 over the weight files in a stated
order. load_artifact verifies the signatures and the hash before it
constructs the model, and raises ValueError on any mismatch, so a
tampered manifest or a tampered weights file is rejected.
"""
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEARN = os.path.join(os.path.dirname(_HERE), "level-1-learn")
if _LEARN not in sys.path:
    sys.path.insert(0, _LEARN)

from signal_ml import SmallNN  # noqa: E402

ARTIFACT_FORMAT = "signal-ml-lab numpy artifact v1 (not SavedModel)"
WEIGHT_FILES = ["W1.npy", "b1.npy", "W2.npy", "b2.npy", "mu.npy", "sd.npy"]
EXPECTED_FEATURE_DIM = 12
EXPECTED_CLASSES = 2


def _weights_sha256(directory):
    h = hashlib.sha256()
    for name in WEIGHT_FILES:
        with open(os.path.join(directory, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def artifact_dir(base_dir, version=1):
    return os.path.join(base_dir, "model", str(int(version)))


def save_artifact(model, mu, sd, base_dir, version=1, config=None):
    """Save a versioned artifact and return its manifest dict."""
    directory = artifact_dir(base_dir, version)
    os.makedirs(directory, exist_ok=True)
    arrays = {
        "W1.npy": model.W1, "b1.npy": model.b1,
        "W2.npy": model.W2, "b2.npy": model.b2,
        "mu.npy": np.asarray(mu), "sd.npy": np.asarray(sd),
    }
    for name in WEIGHT_FILES:
        np.save(os.path.join(directory, name), arrays[name])
    manifest = {
        "format": ARTIFACT_FORMAT,
        "version": int(version),
        "input_signature": {"feature_dim": EXPECTED_FEATURE_DIM, "dtype": "float64"},
        "output_signature": {"classes": EXPECTED_CLASSES,
                             "labels": ["clean", "defect"]},
        "weight_files": list(WEIGHT_FILES),
        "weights_sha256": _weights_sha256(directory),
        "training_config": dict(config or {}),
    }
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_artifact(base_dir, version=1):
    """Verify hash and signatures, then load. Returns (model, mu, sd, manifest)."""
    directory = artifact_dir(base_dir, version)
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        raise ValueError("manifest.json missing in " + directory)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError("unknown artifact format: " + str(manifest.get("format")))
    if int(manifest.get("version", -1)) != int(version):
        raise ValueError("artifact version mismatch")
    input_sig = manifest.get("input_signature", {})
    if int(input_sig.get("feature_dim", -1)) != EXPECTED_FEATURE_DIM:
        raise ValueError("input signature mismatch: feature_dim must be 12")
    output_sig = manifest.get("output_signature", {})
    if int(output_sig.get("classes", -1)) != EXPECTED_CLASSES:
        raise ValueError("output signature mismatch: classes must be 2")
    if list(manifest.get("weight_files", [])) != WEIGHT_FILES:
        raise ValueError("weight file list mismatch")
    for name in WEIGHT_FILES:
        if not os.path.exists(os.path.join(directory, name)):
            raise ValueError("weight file missing: " + name)
    actual = _weights_sha256(directory)
    if actual != manifest.get("weights_sha256"):
        raise ValueError("weights sha256 mismatch: artifact was tampered with or corrupted")
    arrays = {name: np.load(os.path.join(directory, name)) for name in WEIGHT_FILES}
    if arrays["W1.npy"].shape[0] != EXPECTED_FEATURE_DIM:
        raise ValueError("W1 first dimension does not match the input signature")
    if arrays["W2.npy"].shape[1] != EXPECTED_CLASSES:
        raise ValueError("W2 second dimension does not match the output signature")
    model = SmallNN(EXPECTED_FEATURE_DIM,
                    hidden=int(arrays["W1.npy"].shape[1]),
                    n_classes=EXPECTED_CLASSES, seed=0)
    model.W1 = arrays["W1.npy"]
    model.b1 = arrays["b1.npy"]
    model.W2 = arrays["W2.npy"]
    model.b2 = arrays["b2.npy"]
    return model, arrays["mu.npy"], arrays["sd.npy"], manifest


def predict_with_artifact(model, mu, sd, X):
    Xn = (np.asarray(X, dtype=float) - mu) / sd
    return model.predict(Xn)
