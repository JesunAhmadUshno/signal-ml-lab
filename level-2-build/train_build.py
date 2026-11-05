"""Level 2 Build: hand-written training loop at larger scale.

SPEC Build also carried the TensorFlow GradientTape custom loop
forward from Learn. TensorFlow is still not importable in this
environment (verified 2026-11-05, see pipeline.py and README), so the
training loop here is the same hand-written NumPy loop form as Learn:
explicit forward, explicit backward, explicit gradient descent steps
through SmallNN.loss_and_grads and SmallNN.step. No framework fit
call is used anywhere. The network class is imported from the Level 1
Learn code, not copied.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEARN = os.path.join(os.path.dirname(_HERE), "level-1-learn")
if _LEARN not in sys.path:
    sys.path.insert(0, _LEARN)

from signal_ml import SmallNN  # noqa: E402


def train_at_scale(X, y, X_test, y_test, epochs=200, lr=0.05, seed=1,
                   hidden=16, log_every=50):
    """Train SmallNN on pipelined features. Returns (model, metrics, mu, sd)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xn = (X - mu) / sd
    Xtn = (X_test - mu) / sd
    model = SmallNN(X.shape[1], hidden=hidden, n_classes=2, seed=seed)
    loss_curve = []
    for epoch in range(epochs):
        loss, grads = model.loss_and_grads(Xn, y)
        model.step(grads, lr=lr)
        if epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            loss_curve.append({"epoch": int(epoch + 1), "loss": round(float(loss), 4)})
    final_loss, _ = model.loss_and_grads(Xn, y)
    train_pred = model.predict(Xn)
    test_pred = model.predict(Xtn)
    per_class = {}
    for c in (0, 1):
        mask = y_test == c
        per_class[str(c)] = {
            "total": int(np.sum(mask)),
            "correct": int(np.sum(test_pred[mask] == c)),
        }
    metrics = {
        "train_accuracy": round(float(np.mean(train_pred == y)), 4),
        "test_accuracy": round(float(np.mean(test_pred == y_test)), 4),
        "accuracy": round(float(np.mean(test_pred == y_test)), 4),
        "per_class_test": per_class,
        "train_final_loss": round(float(final_loss), 4),
        "loss_curve": loss_curve,
        "epochs": int(epochs),
        "lr": float(lr),
        "hidden": int(hidden),
        "seed": int(seed),
    }
    return model, metrics, mu, sd
