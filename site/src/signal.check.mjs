// Known-answer check, run by `npm run build` before vite: every computation the page does in the
// browser is reproduced here against the repo's own Python outputs and recorded results files.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { features, forward, quantizeModel, score, rocAuc, probe, wilson, RECIPE, LENGTH, PROBE_PER_START } from './signal.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const model = read('./generated/model.json');
const ts = read('./generated/testset.json');
const build = read('../../level-2-build/results_build.json');
const deep = read('../../level-3-deep/results_deep.json');
const rigor = read('../../level-3-deep/results_rigor.json');
const manifest = read('../../level-2-build/artifacts/model/1/manifest.json');

// 0. The recipe the page states and the probe uses is the one in make_dataset.
const py = readFileSync(new URL('../../level-1-learn/signal_ml.py', import.meta.url), 'utf8');
const mk = py.slice(py.indexOf('def make_dataset'), py.indexOf('return X_sig[perm]'));
const num = (re) => mk.match(re).slice(1).map(Number);
assert.deepEqual([...new Set([...mk.matchAll(/rng\.normal\(0, ([\d.]+)/g)].map((m) => +m[1]))], [RECIPE.noise], 'noise');
assert.deepEqual(num(/choice\(\[-1, 1\]\) \* rng\.uniform\(([\d.]+), ([\d.]+)\)/), RECIPE.amp, 'amplitude');
assert.deepEqual(num(/width = int\(rng\.integers\((\d+), (\d+)\)/), RECIPE.width, 'width');
const [s0, sOff] = num(/start = int\(rng\.integers\((\d+), length-(\d+)\)/);
assert.deepEqual([s0, LENGTH - sOff], RECIPE.start, 'start');

assert.equal(model.weights_sha256, manifest.weights_sha256, 'exported weights come from the committed artifact');

// 1. Haar features: JS port vs NumPy extract_features on all 500 signals.
const feats = ts.signals.map((s) => features(s).f);
let maxRel = 0;
feats.forEach((f, i) => f.forEach((v, j) => { const w = ts.py_features[i][j]; maxRel = Math.max(maxRel, Math.abs(v - w) / Math.max(1, Math.abs(w))); }));
assert.ok(maxRel < 1e-12, `feature drift ${maxRel}`);

// 2. Forward pass: identical predictions, scores to float precision.
let maxP = 0;
feats.forEach((f, i) => {
  const r = forward(model, f);
  assert.equal(r.pred, ts.py_pred[i], `prediction ${i}`);
  maxP = Math.max(maxP, Math.abs(r.p[1] - ts.py_p_defect[i]));
});
assert.ok(maxP < 1e-12, `p(defect) drift ${maxP}`);

// 3. Test-set replay reproduces results_build.json.
const s = score(model, feats, ts.labels);
assert.equal(s.correct / 500, build.training.test_accuracy);
assert.equal(s.cm.tn, build.training.per_class_test['0'].correct);
assert.equal(s.cm.tp, build.training.per_class_test['1'].correct);
assert.equal(s.correct, deep.edge_study.float_artifact.correct);

// 4. Int8 toggle reproduces edge.py's quantized predictions and results_deep.json.
const q = quantizeModel(model);
const sq = score(q, feats, ts.labels);
sq.pred.forEach((v, i) => assert.equal(v, ts.py_pred_int8[i], `int8 prediction ${i}`));
assert.equal(sq.correct, deep.edge_study.int8_quantized_numpy_study.correct);

// 5. ROC AUC and Wilson interval match results_rigor.json.
assert.equal(Math.round(rocAuc(s.p, ts.labels) * 1e4) / 1e4, rigor.roc_pr_committed_model.roc_auc);
assert.deepEqual(wilson(s.correct, 500).map((v) => Math.round(v * 1e4) / 1e4), rigor.headline_committed_model.wilson95);

// 6. In-browser difficulty probe (different RNG, same recipe) agrees with the recorded Python sweep.
const sw = rigor.difficulty_sweep;
let worst = 0, total = 0;
for (const c of sw.amp_noise) {
  const r = probe(model, { amp: c.amp, noise: c.noise, n: sw.signals_per_cell, seed: Math.round(c.amp * 100) * 7 + Math.round(c.noise * 100) });
  const d = Math.abs(r.detection - c.detection_rate);
  worst = Math.max(worst, d); total += d;
}
assert.ok(worst < 0.2 && total / sw.amp_noise.length < 0.04, `sweep drift worst ${worst} mean ${total / sw.amp_noise.length}`);
for (const [nz, rec] of Object.entries(sw.false_positive_rate_by_noise)) {
  const r = probe(model, { amp: 1, noise: +nz, n: sw.signals_per_cell, seed: 99 });
  assert.ok(Math.abs(r.fpr - rec) < 0.15, `fpr at noise ${nz}: ${r.fpr} vs ${rec}`);
}
// Haar alignment blind spot: width-4 bumps starting on a multiple of 4 are missed far more often.
const aligned = [], off = [];
for (let st = 0; st <= 60; st++) (st % 4 ? off : aligned).push(probe(model, { amp: RECIPE.amp, noise: RECIPE.noise, n: PROBE_PER_START, seed: 500 + st, width: 4, start: st }).detection);
const avg = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const recAligned = avg(sw.start_width4_at_nominal.filter((x) => x.start % 4 === 0).map((x) => x.detection_rate));
assert.ok(Math.abs(avg(aligned) - recAligned) < 0.12 && avg(off) > 0.95, `alignment ${avg(aligned)} vs ${recAligned}, off ${avg(off)}`);

console.log('signal check ok', { maxFeatureDrift: maxRel, maxProbDrift: maxP, correct: s.correct, int8Correct: sq.correct, sweepWorst: +worst.toFixed(3), alignedJs: +avg(aligned).toFixed(3), alignedPy: +recAligned.toFixed(3) });
