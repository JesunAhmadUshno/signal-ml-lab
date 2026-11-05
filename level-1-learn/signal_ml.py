"""Level 1 Learn: wavelet features plus a small neural network, pure NumPy.

Rebuilds, in teaching form, the published method described in SPEC.md:
wavelet features plus a neural network for fabric defect detection
(Jesun's AIP Conference Proceedings papers, 2023, prior published work).

This code uses synthetic 1-D signals only. It is a teaching stand-in
for fabric sensor or inspection signals, never the original paper data.

Training loop note: SPEC Learn asks for one custom TensorFlow training
loop written by hand past Keras fit. TensorFlow was checked on
2026-11-05 in the local Python and was not importable
(ModuleNotFoundError). Per the honest execution rule for this level,
the same custom hand-written training loop is built here in pure
NumPy: explicit forward, explicit backward, explicit gradient descent
steps, no framework fit call anywhere. The TensorFlow GradientTape
loop is Build scope. See README for the named grade deduction.
"""
import numpy as np

SQRT2 = np.sqrt(2.0)

def haar_dwt(x):
    """One level Haar DWT. x length must be even. Returns (approx, detail)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) % 2 != 0:
        raise ValueError("signal must be 1-D with even length")
    even = x[0::2]; odd = x[1::2]
    approx = (even + odd) / SQRT2
    detail = (even - odd) / SQRT2
    return approx, detail

def haar_idwt(approx, detail):
    approx = np.asarray(approx, dtype=float); detail = np.asarray(detail, dtype=float)
    out = np.empty(approx.shape[0]*2, dtype=float)
    out[0::2] = (approx + detail) / SQRT2
    out[1::2] = (approx - detail) / SQRT2
    return out

def extract_features(signal):
    """12 wavelet features from two Haar levels."""
    a1, d1 = haar_dwt(signal)
    # level 2 on approx (length must be even: 64 -> 32 -> 16 ok)
    a2, d2 = haar_dwt(a1)
    def stats(v):
        return [float(np.mean(v)), float(np.std(v)), float(np.sum(v*v)), float(np.mean(np.abs(v))), float(np.max(np.abs(v)))]
    # Build fixed 12-dim vector
    feats = [
        float(np.mean(a1)), float(np.std(a1)), float(np.sum(a1*a1))/len(signal),
        float(np.mean(d1)), float(np.std(d1)), float(np.sum(d1*d1))/len(signal),
        float(np.max(np.abs(d1))), float(np.mean(np.abs(d1))),
        float(np.sum(a2*a2))/len(signal), float(np.sum(d2*d2))/len(signal),
        float(np.std(a2)), float(np.max(np.abs(d2))),
    ]
    return np.array(feats, dtype=float)

def make_dataset(n_per_class=100, length=64, seed=0):
    """Synthetic signals: label 0 clean, 1 defect (local anomaly). Deterministic."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, length, endpoint=False)
    X_sig = []; y = []
    for i in range(n_per_class):
        base = np.sin(t + rng.uniform(0, 0.3)) + 0.3*np.sin(3*t + rng.uniform(0, 0.3))
        sig = base + rng.normal(0, 0.05, size=length)
        X_sig.append(sig); y.append(0)
    for i in range(n_per_class):
        base = np.sin(t + rng.uniform(0, 0.3)) + 0.3*np.sin(3*t + rng.uniform(0, 0.3))
        sig = base + rng.normal(0, 0.05, size=length)
        start = int(rng.integers(10, length-14))
        width = int(rng.integers(3, 7))
        amp = float(rng.choice([-1, 1]) * rng.uniform(0.8, 1.4))
        sig[start:start+width] += amp
        X_sig.append(sig); y.append(1)
    X_sig = np.array(X_sig); y = np.array(y)
    # deterministic shuffle
    perm = rng.permutation(len(y))
    return X_sig[perm], y[perm]

def features_for_dataset(X_sig):
    return np.array([extract_features(s) for s in X_sig])

class SmallNN:
    """2-layer NN, hand-written forward/backward, softmax cross-entropy."""
    def __init__(self, in_dim, hidden=16, n_classes=2, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0/in_dim), size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0/hidden), size=(hidden, n_classes))
        self.b2 = np.zeros(n_classes)
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h = np.maximum(0, self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        # softmax
        m = np.max(self.z2, axis=1, keepdims=True)
        e = np.exp(self.z2 - m)
        self.p = e / np.sum(e, axis=1, keepdims=True)
        return self.p
    def loss_and_grads(self, X, y):
        n = X.shape[0]
        p = self.forward(X)
        eps = 1e-12
        loss = -np.mean(np.log(p[np.arange(n), y] + eps))
        dz2 = p.copy(); dz2[np.arange(n), y] -= 1; dz2 /= n
        dW2 = self.h.T @ dz2; db2 = np.sum(dz2, axis=0)
        dh = dz2 @ self.W2.T
        dz1 = dh * (self.z1 > 0)
        dW1 = X.T @ dz1; db1 = np.sum(dz1, axis=0)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return float(loss), grads
    def step(self, grads, lr=0.05):
        self.W1 -= lr*grads["W1"]; self.b1 -= lr*grads["b1"]
        self.W2 -= lr*grads["W2"]; self.b2 -= lr*grads["b2"]
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

def train(X, y, X_test, y_test, epochs=200, lr=0.05, seed=1):
    # standardise using train stats
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd==0]=1.0
    Xn = (X-mu)/sd; Xtn = (X_test-mu)/sd
    model = SmallNN(X.shape[1], hidden=16, n_classes=2, seed=seed)
    losses=[]
    for epoch in range(epochs):
        loss, grads = model.loss_and_grads(Xn, y)
        model.step(grads, lr=lr)
        if epoch in (0, 49, 99, 149, epochs-1):
            losses.append({"epoch": epoch+1, "loss": round(float(loss),4)})
    final_loss, _ = model.loss_and_grads(Xn, y)
    pred = model.predict(Xtn)
    acc = float(np.mean(pred==y_test))
    per_class={}
    for c in (0,1):
        mask = y_test==c
        per_class[str(c)] = {"total": int(np.sum(mask)), "correct": int(np.sum(pred[mask]==c))}
    return model, {"accuracy": round(acc,4), "per_class": per_class, "train_final_loss": round(float(final_loss),4), "loss_curve": losses, "mu": mu, "sd": sd}
