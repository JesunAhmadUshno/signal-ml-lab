# Release notes draft: signal-ml-lab

Status: DRAFT notes for releases that already exist. Dated 2026-11-05.
These notes summarise the existing GitHub releases. No new release is
created by this file.

Releases verified with gh release list on 2026-11-05:
v0.1, v0.5, and v1.0, at the URLs below.

## v0.1, Learn: wavelet features plus a hand-written network

URL: https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v0.1

- Haar discrete wavelet transform features (12 features, two Haar
  levels) plus a hand-written NumPy neural network (hidden 16,
  explicit forward, backward, and gradient descent, no framework
  fit call), rebuilt in teaching form from the published method.
- Measured run: synthetic signals only, 200 train and 100 test,
  test accuracy 1.000 (100 of 100, clean 50 of 50, defect 50 of
  50), train loss 0.5467 to a final 0.1037 over 200 epochs.
- Tests: 14 unit tests in tests/test_learn_signal.py, plus the
  scaffold smoke test.
- Grade: 7,300 of 10,000, deductions itemized in the Learn README.
- Limits carried in the release: synthetic signals only,
  TensorFlow not importable (NumPy substitute, named deduction),
  small dataset, Haar features only with no baseline comparison.

## v0.5, Build: pipeline at scale, training, versioned artifact

URL: https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v0.5

- Data pipeline at scale as labelled NumPy and stdlib equivalents
  of the tf.data ideas: parallel map, feature caching, prefetch
  style batching, seeded deterministic batches. Hand-written
  training at larger scale on the Learn features and network,
  imported from Learn, not copied.
- Measured run: 2,900 synthetic signals (2,400 train, 500 test),
  loader speedup 3.247x in the committed run (naive two-pass
  0.4693 seconds, optimized two-pass 0.1445 seconds, timing varies
  between runs), train accuracy 0.9838, test accuracy 0.984 (492
  of 500), train loss 0.5242 to a final 0.0712.
- Artifact: version 1, NumPy artifact format (explicitly not
  SavedModel), manifest with signatures and weights sha256
  93e2f8c113d0d7b625caa1089d23afc6d65ae218d829eae986ee3cb7b3808b21,
  load check passed, tamper cases tested.
- Tests: 20 Build unit tests, 34 in total with Learn.
- Grade: 7,500 of 10,000, deductions itemized in the Build README.
- Limits carried in the release: synthetic signals only, no public
  dataset, TensorFlow equivalents instead of the real framework,
  one two-pass speed workload, single artifact version only.

## v1.0, Deep: inference API and edge deployment study

URL: https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v1.0

- Real-time inference API on stdlib http.server plus NumPy,
  serving the Build artifact through its hash-verified loader:
  GET /health, POST /predict, and bad input returns a clean 400.
  It is a named substitution for TF Serving, not TF Serving.
- Edge deployment study on CPU (CPU by design, no GPU number
  claimed): float artifact 3,530 bytes total (2,896 bytes weight
  files) at accuracy 0.984 (492 of 500); int8 variant as a NumPy
  quantization study, not TFLite, at 1,850 bytes, size ratio
  0.6388, accuracy 0.984 (492 of 500), on the same synthetic test
  set as Build (500 signals, seed 42).
- Latency in the committed run: in-process mean 0.1403 ms over
  100 calls, HTTP mean 1.4033 ms over 50 local calls. Timing
  varies between runs.
- Profiler note: Python cProfile, labelled not TF Profiler.
  Slowest own-code step SmallNN.forward, tottime 0.008043 s,
  share 0.5949 of total tottime.
- Judgment page shipped at level-3-deep/JUDGMENT.md.
- Tests: 9 Deep unit tests, 43 in total with Learn and Build.
- Grade: 7,600 of 10,000, deductions itemized in the Deep README
  (total deductions 2,400).
- TensorFlow was verified not importable (ModuleNotFoundError),
  and no TensorFlow result is claimed by these notes.
- The rest of the v2 scope is the v1.1 backlog in ROADMAP.md. It
  is not part of v1.0 and is not claimed by these notes.
