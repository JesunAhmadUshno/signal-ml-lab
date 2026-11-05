"""Runner: synthetic dataset, wavelet features, hand-written NN training."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from signal_ml import features_for_dataset, make_dataset, train  # noqa: E402


def main():
    Xtr_sig, ytr = make_dataset(n_per_class=100, seed=0)
    Xte_sig, yte = make_dataset(n_per_class=50, seed=42)
    Xtr = features_for_dataset(Xtr_sig)
    Xte = features_for_dataset(Xte_sig)
    _, metrics = train(Xtr, ytr, Xte, yte, epochs=200, lr=0.05, seed=1)
    out = {
        "dataset": (
            "synthetic 1-D signals, labelled synthetic, teaching stand-in for fabric inspection signals, "
            "not original paper data"
        ),
        "train_counts": {"total": int(len(ytr)), "clean": int(np.sum(ytr == 0)), "defect": int(np.sum(ytr == 1))},
        "test_counts": {"total": int(len(yte)), "clean": int(np.sum(yte == 0)), "defect": int(np.sum(yte == 1))},
        "feature_dim": int(Xtr.shape[1]),
        "model": (
            "NumPy SmallNN hidden 16, hand-written forward/backward, gradient descent, no framework fit, "
            "no TensorFlow (not importable 2026-11-05)"
        ),
        "epochs": 200, "lr": 0.05,
        "accuracy": metrics["accuracy"],
        "per_class_test": metrics["per_class"],
        "train_final_loss": metrics["train_final_loss"],
        "loss_curve": metrics["loss_curve"],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_learn.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
