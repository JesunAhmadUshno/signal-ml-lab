# Paper candidate brief: a public replication of a published wavelet method

Project: P17 signal-ml-lab
Status: CANDIDATE, brief written 2026-11-05
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted. The publications
tracker is updated by the coordinator, not by this file.

## Working title

The published method, rebuilt: a public replication of wavelet
features plus a neural network for fabric defect detection, with
pipeline, artifact, serving, and edge measurements

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Numbers below are from synthetic data and are stated as such.
TensorFlow was unavailable in the build environment (import fails
with ModuleNotFoundError), and no TensorFlow result is claimed.

Published methods are rarely rebuilt in public with their
substitutions labelled. This work rebuilds the author's own
published method (AIP Conference Proceedings, 2023: wavelet
features plus a neural network for fabric defect detection) in
three measured levels, on synthetic data only. At Learn level,
Haar discrete wavelet transform features (12 features, two levels)
fed a hand-written NumPy neural network (hidden 16, explicit
forward, backward, and gradient descent, no framework fit call)
to test accuracy 1.000 on 100 synthetic test signals (100 of 100,
clean 50 of 50, defect 50 of 50), with final train loss 0.1037 and
14 unit tests. At Build level, a pipeline with parallel map,
caching, and prefetch style batching trained the same network form
on 2,900 synthetic signals (2,400 train, 500 test), with a loader
speedup of 3.247x in the committed run (naive two-pass 0.4693
seconds, optimized two-pass 0.1445 seconds, timing varies between
runs), train accuracy 0.9838, test accuracy 0.984 (492 of 500),
final train loss 0.0712, and a versioned NumPy artifact format
that is explicitly not SavedModel, with weights sha256
93e2f8c113d0d7b625caa1089d23afc6d65ae218d829eae986ee3cb7b3808b21,
with 20 tests (34 in total). At Deep level, a stdlib http.server
inference API (GET /health, POST /predict, bad input returns 400)
served the Build artifact, and a CPU edge study (CPU by design, no
GPU number claimed) measured the float artifact at 3,530 bytes
total (2,896 bytes of weights) with accuracy 0.984 (492 of 500),
and an int8 variant labelled a NumPy quantization study and not
TFLite at 1,850 bytes, size ratio 0.6388, and accuracy 0.984, with
in-process latency mean 0.1403 ms over 100 calls against HTTP mean
1.4033 ms over 50 calls (timing varies between runs), and a Python
cProfile note labelled not TF Profiler with SmallNN.forward
tottime share 0.5949, with 9 tests (43 in total). The data is
synthetic, the TensorFlow items are named substitutions, and the
paper claims the replication method and its measured numbers on
these sets, not performance on real fabric.

## Method outline

1. Features: hand-written Haar DWT and inverse DWT, with a
   hand-computed check on [4, 6, 10, 12] asserted in tests, and
   extract_features producing 12 features over two Haar levels.
   A defect signal raises detail energy, asserted in tests.
2. Learn network: SmallNN (hidden 16, ReLU, softmax
   cross-entropy) with hand-written gradients, checked against
   central finite differences within 1e-5 in tests. Training is
   explicit gradient descent, 200 epochs, learning rate 0.05.
3. Build pipeline: parallel map over chunks with order preserved,
   an in-memory feature cache, prefetch style batching, and seeded
   deterministic batches covering all 2,400 train rows exactly once.
4. Build training and artifact: full-batch hand-written training
   at scale, then a versioned NumPy artifact (manifest, input and
   output signatures, weight file list, sha256 over the weight
   files), with load checks that reject a tampered manifest or
   weights file. The format is explicitly not SavedModel.
5. Deep API: InferenceService loads the Build artifact through its
   loader (hash verified on load), extracts the Learn Haar
   features, and serves GET /health and POST /predict over stdlib
   http.server. Bad input returns a clean 400 JSON error.
6. Deep edge study: float artifact size on disk, an int8 variant
   built by per-array quantize and dequantize in NumPy (labelled
   a NumPy quantization study, not TFLite), accuracy on the same
   500-signal synthetic test set (seed 42), and in-process versus
   HTTP latency over stated call counts.
7. Profiler note: one inference pass profiled with Python
   cProfile (labelled not TF Profiler), naming the slowest
   own-code step and its tottime share.
8. Judgment: level-3-deep/JUDGMENT.md records when TensorFlow is
   the right tool for this project and when the NumPy path is,
   grounded only in the measured facts above.

## Results table

Measured 2026-11-05, synthetic data only, from the level READMEs
in the repo.

| Level | Set | Headline result | Cost and size | Tests |
| --- | --- | --- | --- | --- |
| Learn | 200 train, 100 test | Accuracy 1.000 (100 of 100), final train loss 0.1037 | 12 features, hidden 16 | 14 |
| Build | 2,400 train, 500 test | Train 0.9838, test 0.984 (492 of 500), final loss 0.0712 | Loader 3.247x committed run (timing varies), artifact v1 not SavedModel | 20 (34 total) |
| Deep, float | 500 test, seed 42 | Accuracy 0.984 (492 of 500) | 3,530 bytes total, 2,896 bytes weights | 9 Deep (43 total) |
| Deep, int8 study | Same set | Accuracy 0.984 (492 of 500) | 1,850 bytes, ratio 0.6388, not TFLite | 9 Deep (43 total) |

Named failure detail: at Build scale all 8 test errors are in the
defect class (clean 250 of 250, defect 242 of 250). At Learn scale
there were no test errors on the smaller set.

## Venue class note

Per SPEC.md, the paper angle for P17 is the rebuild itself: a
public replication of the author's own published method with open
data and labelled substitutions. Venue class is a workshop or
conference in machine learning or signal processing, or a journal
of the AIP or AJSE class, per the publications pipeline tracker
rules. One venue at a time. No venue is named as submitted,
accepted, or contacted in this brief. Fees, if any, arise only
after acceptance and remain Jesun's call under the free-only rule.

## Limits the paper must carry

- Synthetic data only (300, then 2,900 signals). No real fabric
  dataset, no public dataset, and no real-world accuracy claim.
- TensorFlow is unavailable. tf.data, GradientTape, SavedModel,
  TF Serving, TFLite, and TF Profiler are labelled equivalents,
  named substitutions, or v1.1 backlog, never shipped results.
- The loader speedup and all latencies are single-machine timings
  that vary between runs (a second Build run measured 2.692x).
- The int8 variant is a NumPy quantization study: it runs in float
  after dequantization, on one small model and one synthetic set.
- Edge is one Muse VM CPU only. No real edge device, no GPU
  number, no load or concurrency measurement.
- The v1.1 backlog (real TensorFlow and TFLite conversion, a real
  public fabric dataset run, TF Serving, larger edge matrix) is
  future work, named in ROADMAP.md in the repo.

## Submission gate

Submission is Jesun's tap and nothing is submitted. This brief is
the candidate only.
