# Level 1, Learn: signal-ml-lab

Status: DONE 2026-11-05 (ROADMAP-V3-BATCH, Sprint 2 early pull). Evidence: code in this folder, 14 unit tests in tests/test_learn_signal.py, and measured results below. Build and Deep are not started.

## How this maps to the published method

Prior published work (separate from this run): Jesun's AIP Conference Proceedings papers, 2023, described in PROPOSAL.md (former SPEC.md and DOSSIER.md) as wavelet features plus a neural network for fabric defect detection. No original paper numbers are quoted here as this run's results, and no original paper data is used.

This Learn level rebuilds that idea in teaching form:
1. A signal arrives as a 1-D sequence.
2. A hand-written Haar discrete wavelet transform splits it into approximation and detail coefficients.
3. Energy and summary features are computed from those coefficients. A local defect raises detail energy, which a reader can check in a test.
4. A small neural network, written by hand in NumPy, learns clean versus defect from those 12 features.

## TensorFlow status, honest

SPEC Learn asks for one custom TensorFlow training loop written by hand past Keras fit.
Check on 2026-11-05: `python3 -c "import tensorflow"` in the local Python returned ModuleNotFoundError: No module named 'tensorflow'. TensorFlow was NOT available.
No large package was installed silently and no TensorFlow run was faked.
Instead the same custom hand-written training loop was built in pure NumPy: explicit forward, explicit backward, explicit gradient descent steps, no framework fit call.
The TensorFlow GradientTape loop is Build scope (see REPORT.md, Remaining gaps).

## What is in the code

- signal_ml.py: haar_dwt and haar_idwt, extract_features (12 features, two Haar levels), make_dataset (synthetic, seeded, deterministic), SmallNN (hidden 16, ReLU, softmax cross-entropy, hand-written gradients), train.
- run_learn.py: builds train (200) and test (100) synthetic sets, extracts features, trains 200 epochs at lr 0.05, writes results_learn.json. Two runs produced identical results_learn.json (verified 2026-11-05).

## Synthetic data, labelled synthetic

Signals are length 64. Clean signals are sine mixtures plus small noise. Defect signals are the same plus a local anomaly (a short spike or dip). This is a teaching stand-in for fabric sensor or inspection signals. It is never presented as real fabric data or as the original paper data. The perfect test score below reflects how separable this synthetic anomaly is, not real-world fabric performance.

## Run it

```text
pip install numpy
python -m unittest discover -s tests -p "test_learn*.py" -v
python level-1-learn/run_learn.py
python tests/test_smoke.py
```

## Measured results (2026-11-05, Muse VM, Python 3.12, NumPy 1.26.4)

Train: 200 signals (100 clean, 100 defect). Test: 100 signals (50 clean, 50 defect). Feature dim 12.

| Metric | Measured |
| --- | --- |
| Test accuracy | 1.000 (100 of 100) |
| Clean correct | 50 of 50 |
| Defect correct | 50 of 50 |
| Train loss epoch 1 | 0.5467 |
| Train loss epoch 50 | 0.1788 |
| Train loss epoch 100 | 0.1366 |
| Train loss epoch 150 | 0.1169 |
| Train loss epoch 200 (curve) | 0.1039 |
| Train final loss | 0.1037 |

Hand-computed DWT check: for [4, 6, 10, 12], approx is [7.0710678, 15.5563492] and detail is [-1.4142136, -1.4142136], asserted in tests. Gradient check: analytic W2 gradient agrees with central finite differences within 1e-5 in tests.

## Limits

- Synthetic only, small, one seed set for the headline run.
- Accuracy 1.000 is on synthetic data only.
- No TensorFlow, no real fabric dataset, no public dataset pipeline: those are Build scope.

## Next

Level 2, Build: TensorFlow GradientTape custom loop and tf.data pipeline on a public dataset per SPEC, plus this NumPy loop kept as the teaching reference.
