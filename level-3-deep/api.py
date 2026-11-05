"""Level 3 Deep: real-time inference API, stdlib http.server plus NumPy only.

Loads the shipped Level 2 Build artifact (level-2-build/artifacts/model/1)
through the Build artifact loader (level-2-build/artifact.py, imported, not
copied). Features are the same Haar wavelet features as Learn and Build:
extract_features is imported from level-1-learn/signal_ml.py, not copied.

Endpoints:
- GET /health returns {"status": "ok", "artifact_version": 1}
- POST /predict takes a raw signal as a JSON list of numbers, or as
  {"signal": [numbers]}, and returns the class, label, and score.
  Bad input returns a clean 400 JSON error, never a stack trace.

TensorFlow status: TensorFlow was verified not importable on
2026-11-05 (ModuleNotFoundError: No module named 'tensorflow').
TensorFlow Serving is therefore a named substitution, not shipped:
this API serves the NumPy artifact with the standard library only.
No TensorFlow result is claimed anywhere in this level.
"""
import importlib.util
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEARN = os.path.join(_ROOT, "level-1-learn")
_BUILD = os.path.join(_ROOT, "level-2-build")
for _p in (_LEARN, _BUILD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artifact import load_artifact  # noqa: E402
from signal_ml import extract_features  # noqa: E402

LABELS = ["clean", "defect"]
EXPECTED_SIGNAL_LENGTH = 64
DEFAULT_ARTIFACT_BASE = os.path.join(_BUILD, "artifacts")


def tensorflow_available():
    return importlib.util.find_spec("tensorflow") is not None


class InferenceService:
    """In-process inference over the Build artifact."""

    def __init__(self, artifact_base=None, version=1):
        base = artifact_base or DEFAULT_ARTIFACT_BASE
        self.model, self.mu, self.sd, self.manifest = load_artifact(base, version=version)
        self.version = int(version)
        self.artifact_base = base

    def predict_signal(self, signal):
        """Predict one raw signal. Raises ValueError on bad input."""
        arr = validate_signal(signal)
        feats = extract_features(arr)
        xn = (feats - self.mu) / self.sd
        probs = self.model.forward(xn.reshape(1, -1))[0]
        cls = int(np.argmax(probs))
        return {
            "class": cls,
            "label": LABELS[cls],
            "score": round(float(probs[cls]), 6),
            "probabilities": [round(float(p), 6) for p in probs],
        }

    def predict_features(self, feats):
        xn = (np.asarray(feats, dtype=float) - self.mu) / self.sd
        probs = self.model.forward(np.atleast_2d(xn))
        cls = np.argmax(probs, axis=1)
        return cls, probs


def validate_signal(signal):
    if isinstance(signal, np.ndarray):
        signal = signal.tolist()
    if not isinstance(signal, (list, tuple)):
        raise ValueError("signal must be a JSON list of numbers")
    if len(signal) != EXPECTED_SIGNAL_LENGTH:
        raise ValueError("signal length must be 64, got " + str(len(signal)))
    try:
        arr = np.asarray(signal, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("signal must contain only numbers")
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError("signal must be a finite 1-D list of numbers")
    return arr


def parse_predict_body(raw):
    """Parse a POST body: a JSON list, or an object with a signal field."""
    try:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception:
        raise ValueError("body must be valid JSON")
    if isinstance(data, dict):
        if "signal" not in data:
            raise ValueError("JSON object body must contain a signal field")
        data = data["signal"]
    return validate_signal(data), data


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/"):
                self._send_json(200, {"status": "ok", "artifact_version": service.version,
                                      "labels": LABELS})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                arr, _original = parse_predict_body(raw)
                result = service.predict_signal(arr)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception:
                self._send_json(400, {"error": "bad input"})
                return
            self._send_json(200, result)

    return Handler


def serve(service=None, host="127.0.0.1", port=8765):
    service = service or InferenceService()
    server = ThreadingHTTPServer((host, port), make_handler(service))
    return server
