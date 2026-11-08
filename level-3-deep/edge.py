"""Level 3 Deep: edge deployment study on CPU.

Edge is CPU by design in this study. No GPU number is claimed and the
assigned BlackBox GPU node was not used, as the level README states. All timing is on the Muse VM CPU.

Parts:
1. Model size on disk for the float artifact (the shipped Build
   artifact files under level-2-build/artifacts/model/1).
2. An int8-quantized weight variant built in code: weight arrays are
   quantized to int8 with a per-array scale and dequantized back to
   float for inference. This is labelled a NumPy quantization study,
   not TFLite. TFLite is not shipped, because TensorFlow was verified
   not importable on 2026-11-05 (ModuleNotFoundError). The quantized
   size, the accuracy on the same synthetic test set as Build, and
   the latency per call are measured.
3. In-process versus HTTP latency comparison: mean and p95 over a
   stated number of calls. Timing fields are measured and will vary:
   they are the named non-deterministic fields of results_deep.json.

The synthetic test set is the same as Build: make_dataset with
n_per_class 250 and seed 42, 500 signals total, generated through the
Build data generation path (level-2-build/pipeline.make_large_dataset,
which calls the Learn generator). Synthetic data only.
"""
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEARN = os.path.join(_ROOT, "level-1-learn")
_BUILD = os.path.join(_ROOT, "level-2-build")
for _p in (_LEARN, _BUILD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artifact import WEIGHT_FILES, artifact_dir  # noqa: E402
from pipeline import make_large_dataset  # noqa: E402
from signal_ml import SmallNN, extract_features  # noqa: E402

TEST_PER_CLASS = 250
TEST_SEED = 42


def artifact_size_bytes(artifact_base=None, version=1):
    base = artifact_base or os.path.join(_BUILD, "artifacts")
    directory = artifact_dir(base, version)
    per_file = {}
    total = 0
    for name in sorted(os.listdir(directory)):
        size = int(os.path.getsize(os.path.join(directory, name)))
        per_file[name] = size
        total += size
    weight_total = int(sum(per_file[n] for n in WEIGHT_FILES))
    return {"total_bytes": total, "weight_files_bytes": weight_total, "per_file_bytes": per_file}


def _saved_npy_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return len(buf.getvalue())


def quantize_array(arr):
    arr = np.asarray(arr, dtype=float)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, np.float64(scale)


def dequantize_array(q, scale):
    return np.asarray(q, dtype=np.int8).astype(float) * float(scale)


def build_quantized_model(service):
    """Return (model, mu, sd, quantized_size_bytes, float_array_bytes).

    Quantized size counts the int8 arrays as saved .npy bytes plus one
    float64 scale per array, for the six artifact arrays (W1, b1, W2,
    b2, mu, sd). Float array bytes are the same six arrays as saved
    float .npy bytes, so the two sizes compare like for like.
    """
    arrays = {"W1": service.model.W1, "b1": service.model.b1,
              "W2": service.model.W2, "b2": service.model.b2,
              "mu": service.mu, "sd": service.sd}
    deq = {}
    q_size = 0
    float_size = 0
    for name, arr in arrays.items():
        q, scale = quantize_array(arr)
        q_size += _saved_npy_bytes(q) + _saved_npy_bytes(np.array(scale))
        float_size += _saved_npy_bytes(np.asarray(arr, dtype=float))
        deq[name] = dequantize_array(q, scale)
    model = SmallNN(12, hidden=int(deq["W1"].shape[1]), n_classes=2, seed=0)
    model.W1, model.b1, model.W2, model.b2 = deq["W1"], deq["b1"], deq["W2"], deq["b2"]
    return model, deq["mu"], deq["sd"], int(q_size), int(float_size)


def load_test_set():
    """Same synthetic test set as Build: 500 signals, seed 42."""
    signals, labels = make_large_dataset(n_per_class=TEST_PER_CLASS, seed=TEST_SEED)
    feats = np.array([extract_features(s) for s in signals])
    return signals, labels, feats


def accuracy_for(model, mu, sd, feats, labels):
    xn = (np.asarray(feats, dtype=float) - mu) / sd
    pred = model.predict(xn)
    return round(float(np.mean(pred == labels)), 4), int(np.sum(pred == labels))


def _latency_stats(times_ms):
    arr = np.array(times_ms, dtype=float)
    return {"mean_ms": round(float(np.mean(arr)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "calls": int(len(arr))}


def measure_inprocess_latency(service, signal, calls=100):
    # Warm up once so the first-call cost does not skew the sample.
    service.predict_signal(signal)
    times = []
    for _ in range(int(calls)):
        t0 = time.perf_counter()
        service.predict_signal(signal)
        times.append((time.perf_counter() - t0) * 1000.0)
    return _latency_stats(times)


def measure_quantized_latency(model, mu, sd, signal, calls=100):
    feats = extract_features(np.asarray(signal, dtype=float))

    def one():
        xn = (feats - mu) / sd
        return model.predict(xn.reshape(1, -1))

    one()
    times = []
    for _ in range(int(calls)):
        t0 = time.perf_counter()
        one()
        times.append((time.perf_counter() - t0) * 1000.0)
    stats = _latency_stats(times)
    # Also report a full per-call figure including feature extraction.
    full = []
    for _ in range(int(calls)):
        t0 = time.perf_counter()
        f = extract_features(np.asarray(signal, dtype=float))
        xn = (f - mu) / sd
        model.predict(xn.reshape(1, -1))
        full.append((time.perf_counter() - t0) * 1000.0)
    stats["mean_ms_with_features"] = round(float(np.mean(np.array(full))), 4)
    return stats


def measure_http_latency(base_url, signal, calls=50):
    payload = json.dumps({"signal": [float(v) for v in signal]}).encode("utf-8")

    def one():
        req = urllib.request.Request(base_url.rstrip("/") + "/predict", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    one()
    times = []
    last = None
    for _ in range(int(calls)):
        t0 = time.perf_counter()
        last = one()
        times.append((time.perf_counter() - t0) * 1000.0)
    stats = _latency_stats(times)
    stats["last_response"] = last
    return stats


def run_edge_study(service, base_url=None, latency_calls=100, http_calls=50):
    signals, labels, feats = load_test_set()
    float_acc, float_correct = accuracy_for(service.model, service.mu, service.sd, feats, labels)
    sizes = artifact_size_bytes(service.artifact_base, service.version)
    q_model, q_mu, q_sd, q_size, float_array_bytes = build_quantized_model(service)
    q_acc, q_correct = accuracy_for(q_model, q_mu, q_sd, feats, labels)
    sample = signals[0]
    inproc = measure_inprocess_latency(service, sample, calls=latency_calls)
    q_lat = measure_quantized_latency(q_model, q_mu, q_sd, sample, calls=latency_calls)
    result = {
        "test_set": {"total": int(len(labels)), "per_class": int(TEST_PER_CLASS),
                     "seed": int(TEST_SEED), "source": "Build make_large_dataset, synthetic only"},
        "float_artifact": {"size_on_disk": sizes, "accuracy": float_acc,
                           "correct": float_correct},
        "int8_quantized_numpy_study": {
            "label": "NumPy quantization study, not TFLite",
            "quantized_size_bytes": int(q_size),
            "float_array_bytes_comparable": int(float_array_bytes),
            "size_ratio_quantized_over_float": round(float(q_size) / float(float_array_bytes), 4)
            if float_array_bytes else None,
            "accuracy": q_acc, "correct": q_correct,
            "latency": q_lat,
        },
        "latency_inprocess": inproc,
    }
    if base_url:
        result["latency_http"] = measure_http_latency(base_url, sample, calls=http_calls)
        if result["latency_http"]["mean_ms"] and inproc["mean_ms"]:
            result["http_overhead_ratio_mean"] = round(
                float(result["latency_http"]["mean_ms"]) / float(inproc["mean_ms"]), 3)
    return result
