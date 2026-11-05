"""Level 2 Build runner.

Runs both loaders over a larger synthetic set with timing, trains the
hand-written network on the pipelined features, saves and load-checks
a versioned NumPy artifact, and writes results_build.json.

Timing honesty: the timing workload is two full passes over the
training signals. The naive loader extracts features on both passes.
The optimized pipeline extracts once in parallel and serves the
second pass from cache, which is exactly the tf.data caching benefit
this level demonstrates. Timing values live under measured_timing and
are the only fields expected to differ between runs; every other
field is deterministic for the stated seeds.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from artifact import load_artifact, predict_with_artifact, save_artifact  # noqa: E402
from pipeline import OptimizedPipeline, PrefetchBatches, make_large_dataset, naive_features  # noqa: E402
from train_build import train_at_scale  # noqa: E402

TRAIN_PER_CLASS = 1200
TEST_PER_CLASS = 250
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.05


def tensorflow_available():
    return importlib.util.find_spec("tensorflow") is not None


def main():
    Xtr_sig, ytr = make_large_dataset(n_per_class=TRAIN_PER_CLASS, seed=0)
    Xte_sig, yte = make_large_dataset(n_per_class=TEST_PER_CLASS, seed=42)

    # Naive loader: two full passes, extraction both times.
    t0 = time.perf_counter()
    naive_first = naive_features(Xtr_sig)
    _ = naive_features(Xtr_sig)
    naive_two_pass = time.perf_counter() - t0

    # Optimized pipeline: parallel extraction once, second pass cached.
    pipe = OptimizedPipeline(Xtr_sig, ytr, workers=4, chunk_size=256, seed=123)
    t0 = time.perf_counter()
    opt_first, _ = pipe.load()
    opt_second, _ = pipe.load()
    optimized_two_pass = time.perf_counter() - t0

    features_equal = bool(np.array_equal(naive_first, opt_first)
                          and np.array_equal(opt_first, opt_second))
    if not features_equal:
        raise RuntimeError("naive and optimized features differ")

    test_pipe = OptimizedPipeline(Xte_sig, yte, workers=4, chunk_size=256, seed=123)
    Xte, _ = test_pipe.load()

    batches = list(PrefetchBatches(pipe, batch_size=BATCH_SIZE, shuffle=True))
    batch_total = int(sum(len(labels) for _, labels in batches))
    if batch_total != int(len(ytr)):
        raise RuntimeError("prefetch batches do not cover the training set")

    config = {"epochs": EPOCHS, "lr": LR, "hidden": 16, "seed": 1,
              "train_signals": int(len(ytr)), "feature_dim": 12,
              "data": "synthetic only, seeded"}
    model, metrics, mu, sd = train_at_scale(
        opt_first, ytr, Xte, yte, epochs=EPOCHS, lr=LR, seed=1)

    artifacts_base = os.path.join(_HERE, "artifacts")
    manifest = save_artifact(model, mu, sd, artifacts_base, version=1, config=config)
    loaded_model, loaded_mu, loaded_sd, loaded_manifest = load_artifact(artifacts_base, version=1)
    loaded_pred = predict_with_artifact(loaded_model, loaded_mu, loaded_sd, Xte)
    direct_pred = model.predict((Xte - mu) / sd)
    load_check = bool(np.array_equal(loaded_pred, direct_pred))
    if not load_check:
        raise RuntimeError("loaded artifact predictions differ from the trained model")
    loaded_accuracy = round(float(np.mean(loaded_pred == yte)), 4)

    speedup = round(naive_two_pass / optimized_two_pass, 3) if optimized_two_pass > 0 else None
    out = {
        "dataset": (
            "synthetic 1-D signals, labelled synthetic, teaching stand-in for fabric inspection signals, "
            "not original paper data, not a public dataset"
        ),
        "train_counts": {"total": int(len(ytr)), "clean": int(np.sum(ytr == 0)), "defect": int(np.sum(ytr == 1))},
        "test_counts": {"total": int(len(yte)), "clean": int(np.sum(yte == 0)), "defect": int(np.sum(yte == 1))},
        "total_signals": int(len(ytr) + len(yte)),
        "feature_dim": int(opt_first.shape[1]),
        "pipeline": {
            "workers": 4, "chunk_size": 256, "batch_size": BATCH_SIZE,
            "batch_seed": 123, "batch_count": int(len(batches)),
            "batch_total_rows": batch_total,
            "extraction_count_first_pipeline": int(pipe.extraction_count),
            "cache_hits": int(pipe.cache_hits), "cache_misses": int(pipe.cache_misses),
            "features_equal_naive_vs_optimized": features_equal,
            "ideas": (
                "tf.data parallel map, caching and prefetch implemented as labelled NumPy and stdlib equivalents; "
                "TensorFlow not available"
            ),
        },
        "measured_timing": {
            "workload": (
                "two full passes over the training signals; naive extracts both passes, optimized extracts once "
                "in parallel and serves the second pass from cache"
            ),
            "naive_two_pass_seconds": round(float(naive_two_pass), 4),
            "optimized_two_pass_seconds": round(float(optimized_two_pass), 4),
            "speedup_two_pass": speedup,
            "measured_on": "Muse VM, run to run variance expected",
        },
        "training": metrics,
        "artifact": {
            "format": manifest["format"],
            "versions": [1],
            "weights_sha256": manifest["weights_sha256"],
            "input_signature": manifest["input_signature"],
            "output_signature": manifest["output_signature"],
            "load_check_passed": load_check,
            "loaded_test_accuracy": loaded_accuracy,
        },
        "environment": {
            "numpy_version": np.__version__,
            "tensorflow_available": bool(tensorflow_available()),
            "note": (
                "python3 -c import tensorflow failed with ModuleNotFoundError on 2026-11-05; torch also absent; "
                "no package installed"
            ),
        },
    }
    path = os.path.join(_HERE, "results_build.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
