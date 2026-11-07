// JavaScript port of the repo's NumPy maths, used live in the browser.
// haar / features: level-1-learn/signal_ml.py (haar_dwt, extract_features)
// forward: SmallNN.forward (ReLU hidden layer, softmax output)
// quantize: level-3-deep/edge.py (quantize_array / dequantize_array, per-array max|w|/127)
// makeSignal: the make_dataset recipe. The browser RNG is not NumPy's PCG64, so live signals
// follow the same recipe but are different samples. src/signal.check.mjs asserts parity on
// the exported seed-42 test set during `npm run build`.

const SQRT2 = Math.sqrt(2);
export const LENGTH = 64;

export function haar(x) {
  const n = x.length / 2, a = new Array(n), d = new Array(n);
  for (let i = 0; i < n; i++) { a[i] = (x[2 * i] + x[2 * i + 1]) / SQRT2; d[i] = (x[2 * i] - x[2 * i + 1]) / SQRT2; }
  return [a, d];
}

const sum = (v) => v.reduce((s, x) => s + x, 0);
const mean = (v) => sum(v) / v.length;
const std = (v) => { const m = mean(v); return Math.sqrt(mean(v.map((x) => (x - m) * (x - m)))); }; // np.std, ddof 0
const energy = (v) => sum(v.map((x) => x * x));
const maxAbs = (v) => Math.max(...v.map(Math.abs));

export const FEATURE_NAMES = [
  'mean a1', 'std a1', 'energy a1', 'mean d1', 'std d1', 'energy d1',
  'max |d1|', 'mean |d1|', 'energy a2', 'energy d2', 'std a2', 'max |d2|',
];

export function features(sig) {
  const [a1, d1] = haar(sig);
  const [a2, d2] = haar(a1);
  const L = sig.length;
  return {
    coeffs: { a1, d1, a2, d2 },
    f: [mean(a1), std(a1), energy(a1) / L, mean(d1), std(d1), energy(d1) / L,
      maxAbs(d1), mean(d1.map(Math.abs)), energy(a2) / L, energy(d2) / L, std(a2), maxAbs(d2)],
  };
}

export function forward(m, f) {
  const x = f.map((v, j) => (v - m.mu[j]) / m.sd[j]);
  const H = m.b1.length;
  const h = new Array(H);
  for (let k = 0; k < H; k++) { let z = m.b1[k]; for (let j = 0; j < x.length; j++) z += x[j] * m.W1[j][k]; h[k] = Math.max(0, z); }
  const z2 = m.b2.map((b, c) => { let z = b; for (let k = 0; k < H; k++) z += h[k] * m.W2[k][c]; return z; });
  const mx = Math.max(...z2), e = z2.map((z) => Math.exp(z - mx)), s = sum(e);
  const p = e.map((v) => v / s);
  return { x, p, pred: p[1] > p[0] ? 1 : 0 };
}

// np.round rounds half to even; keep that so int8 codes match edge.py exactly.
const roundHalfEven = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };
function quantizeArray(arr) {
  const flat = arr.flat();
  const mx = Math.max(...flat.map(Math.abs));
  const scale = mx > 0 ? mx / 127 : 1;
  const deq = (v) => Math.min(127, Math.max(-127, roundHalfEven(v / scale))) * scale;
  return Array.isArray(arr[0]) ? arr.map((row) => row.map(deq)) : arr.map(deq);
}
export function quantizeModel(m) {
  const out = {};
  for (const k of ['W1', 'b1', 'W2', 'b2', 'mu', 'sd']) out[k] = quantizeArray(m[k]);
  return out;
}

export function score(m, feats, labels, threshold = 0.5) {
  const p = feats.map((f) => forward(m, f).p[1]);
  const pred = p.map((v) => (v > threshold ? 1 : 0));
  const cm = { tn: 0, fp: 0, fn: 0, tp: 0 };
  labels.forEach((y, i) => { cm[y ? (pred[i] ? 'tp' : 'fn') : (pred[i] ? 'fp' : 'tn')]++; });
  return { p, pred, cm, correct: cm.tn + cm.tp };
}

export function rocAuc(p, labels) {
  // Rank-based AUC (probability a random defect scores above a random clean signal).
  let auc = 0, P = 0, N = 0;
  const pos = [], neg = [];
  labels.forEach((y, i) => (y ? pos : neg).push(p[i]));
  P = pos.length; N = neg.length;
  for (const a of pos) for (const b of neg) auc += a > b ? 1 : a === b ? 0.5 : 0;
  return auc / (P * N);
}

export function rng(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = () => { let a = 0; while (a === 0) a = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u()); };
  const int = (lo, hi) => lo + Math.floor(u() * (hi - lo)); // half-open, like rng.integers
  return { u, normal, int };
}

// One signal from the make_dataset recipe. defect: null or { start, width, amp }.
export function makeSignal(r, { noise = 0.05, defect = null } = {}) {
  const p1 = r.u() * 0.3, p3 = r.u() * 0.3;
  const sig = new Array(LENGTH);
  for (let i = 0; i < LENGTH; i++) {
    const t = (2 * Math.PI * i) / LENGTH;
    sig[i] = Math.sin(t + p1) + 0.3 * Math.sin(3 * t + p3) + noise * r.normal();
  }
  if (defect) for (let i = defect.start; i < Math.min(LENGTH, defect.start + defect.width); i++) sig[i] += defect.amp;
  return sig;
}

// Detection rate under the model for n defect signals, plus false-positive rate on n clean ones.
// amp is the absolute amplitude or a [lo, hi] range (random sign); width / start are fixed or drawn as in training.
export function probe(m, { amp, noise, n = 200, seed = 1, width = null, start = null }) {
  const r = rng(seed);
  let hits = 0, fps = 0;
  for (let i = 0; i < n; i++) {
    const w = width ?? r.int(3, 7), s = start ?? r.int(10, LENGTH - 14), sign = r.u() < 0.5 ? -1 : 1;
    const a = Array.isArray(amp) ? amp[0] + r.u() * (amp[1] - amp[0]) : amp;
    if (forward(m, features(makeSignal(r, { noise, defect: { start: s, width: w, amp: sign * a } })).f).pred) hits++;
    if (forward(m, features(makeSignal(r, { noise })).f).pred) fps++;
  }
  return { detection: hits / n, fpr: fps / n, n };
}

export function wilson(k, n, z = 1.959964) {
  const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [c - h, c + h];
}
