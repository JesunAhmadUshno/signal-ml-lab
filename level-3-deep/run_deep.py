"""Level 3 Deep runner.

Runs the whole Deep study end to end:
- verifies the TensorFlow absence honestly (named substitution)
- loads the Build artifact through the API path
- profiles one inference pass with Python cProfile (labelled a Python
  profiler, not TF Profiler)
- runs the edge study (sizes, int8 NumPy quantization study, accuracy)
- starts the API on a local port, measures HTTP latency against
  in-process latency, and shuts the API down
- writes level-3-deep/results_deep.json

Deterministic except the named timing fields: every value under
measured_timing and every latency block will vary between runs. All
other fields are deterministic for the stated seeds.
"""
import cProfile
import importlib.util
import io
import json
import os
import pstats
import sys
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from api import InferenceService, serve  # noqa: E402
from edge import load_test_set, run_edge_study  # noqa: E402

LATENCY_CALLS = 100
HTTP_CALLS = 50
PORT = 18765


def tensorflow_available():
    return importlib.util.find_spec("tensorflow") is not None


def profile_inference(service):
    """cProfile on one inference pass over the synthetic test features.

    Labelled a Python profiler note, not TF Profiler: TensorFlow is not
    available in this environment, so no TF Profiler result exists.
    """
    _signals, _labels, feats = load_test_set()
    xn = (feats - service.mu) / service.sd
    profiler = cProfile.Profile()
    profiler.enable()
    for row in feats:
        service.model.forward(((row - service.mu) / service.sd).reshape(1, -1))
    _ = service.model.predict(xn)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    total = 0.0
    entries = []
    for func, (_cc, _nc, tt, ct, _callers) in stats.stats.items():
        total += tt
        entries.append({"function": func[2], "file": os.path.basename(func[0]),
                        "line": int(func[1]), "tottime_s": round(float(tt), 6),
                        "cumtime_s": round(float(ct), 6), "calls": int(_nc)})
    entries.sort(key=lambda e: e["cumtime_s"], reverse=True)
    top = entries[:8]
    # Share of the slowest named own-code step against total tottime.
    own = [e for e in entries if e["file"] in ("signal_ml.py", "api.py", "edge.py")]
    own_slowest = own[0] if own else (top[0] if top else {})
    own_share = round(float(own_slowest.get("tottime_s", 0.0)) / total, 4) if total > 0 else 0.0
    return {
        "profiler": "Python cProfile, not TF Profiler",
        "workload": "one inference pass: per-signal forward over 500 test feature rows plus one batch predict",
        "total_tottime_s": round(float(total), 6),
        "slowest_step": own_slowest,
        "slowest_step_tottime_share": own_share,
        "top_steps": top,
        "note": (
            "Timing inside the profiler is measured and will vary. The slowest own-code step is named "
            "from this run in the level README."
        ),
    }


def main():
    service = InferenceService()
    profile = profile_inference(service)

    server = serve(service, host="127.0.0.1", port=PORT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = "http://127.0.0.1:" + str(PORT)
    try:
        edge = run_edge_study(service, base_url=base_url,
                              latency_calls=LATENCY_CALLS, http_calls=HTTP_CALLS)
    finally:
        server.shutdown()
        server.server_close()

    out = {
        "level": "Level 3 Deep, signal-ml-lab (P17)",
        "date": "2026-11-05",
        "dataset": "synthetic only, same synthetic test set as Build: 500 signals, seed 42",
        "environment": {
            "numpy_version": np.__version__,
            "tensorflow_available": bool(tensorflow_available()),
            "note": (
                "python3 -c import tensorflow failed with ModuleNotFoundError on 2026-11-05; "
                "no package installed, no TensorFlow result claimed"
            ),
            "hardware": "Muse VM CPU only, edge is CPU by design, BlackBox GPU node not used",
        },
        "api": {
            "server": "stdlib http.server ThreadingHTTPServer plus NumPy",
            "artifact": "level-2-build/artifacts/model/1, loaded through Build artifact.load_artifact",
            "endpoints": ["GET /health", "POST /predict"],
            "signal_length": 64,
            "bad_input_status": 400,
        },
        "tensorflow_substitutions": {
            "tensorflow_serving": "named substitution: stdlib HTTP API serving the NumPy artifact, not TF Serving",
            "tflite": "named substitution: NumPy int8 quantization study in edge.py, not TFLite",
            "tf_profiler": "named substitution: Python cProfile note, not TF Profiler",
        },
        "edge_study": {
            "test_set": edge["test_set"],
            "float_artifact": edge["float_artifact"],
            "int8_quantized_numpy_study": edge["int8_quantized_numpy_study"],
        },
        "measured_timing": {
            "note": "Non-deterministic fields: all latency values vary between runs.",
            "latency_calls_inprocess": LATENCY_CALLS,
            "latency_calls_http": HTTP_CALLS,
            "latency_inprocess": edge["latency_inprocess"],
            "latency_http": edge.get("latency_http"),
            "http_overhead_ratio_mean": edge.get("http_overhead_ratio_mean"),
        },
        "profiler_note": profile,
    }
    path = os.path.join(_HERE, "results_deep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
