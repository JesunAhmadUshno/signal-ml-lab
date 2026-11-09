# The Published Method, Rebuilt: Building signal-ml-lab in the Open

By Jesun Ahmad Ushno
Date: 2026-11-05
Status: DRAFT, written fan-out for P17 signal-ml-lab v1.0. Not posted.

## The problem

A published method that nobody can run is hard to check, teach, or
extend. This project rebuilds, clean in public, the wavelet method
from Jesun's own AIP Conference Proceedings papers (2023): Haar
wavelet features plus a neural network for fabric defect detection.
The rebuild is honest about what it is. All data in v1.0 is
synthetic, labelled synthetic, and is a teaching stand-in for fabric
inspection signals, not the original paper data and not real fabric
data. TensorFlow is unavailable in the build environment, and no
TensorFlow result is claimed anywhere in this project.

This project, P17 in Roadmap v3, builds the method in three levels:
wavelet features and a hand-written network, a data pipeline and
training at larger scale with a versioned artifact, then a real-time
inference API and a CPU edge deployment study.

Repository: https://github.com/JesunAhmadUshno/signal-ml-lab

## Past, present, future (the dossier, condensed)

Past: the original published method (Jesun's own AIP papers, 2023)
used wavelet features plus a neural network for fabric defect
detection. The prior art and the problem history are recorded in
DOSSIER.md and NOVELTY.md in the repo. The novelty claim there stays
marked PROPOSED. This post makes no novelty claim beyond the
measured work below, and it quotes no original paper number as a
result of this run.

Present: v1.0 closes three levels on 2026-11-05. Learn rebuilt the
method in teaching form with Haar features and a hand-written NumPy
network. Build scaled it to 2,900 synthetic
signals with a pipeline speed table and a versioned NumPy artifact. Deep added a stdlib HTTP inference API and
a CPU edge study with float and int8 sizes, accuracies, and
latencies.

Future: the rest of the v2 scope is named in ROADMAP.md in the repo
as the v1.1 backlog. It includes a real TensorFlow and TFLite
conversion, a GradientTape loop and a tf.data pipeline, real
SavedModel discipline, TF Serving deployment, a TF Profiler note,
a real public fabric dataset run, and a larger edge hardware
matrix. None of that is done, and none of it is claimed here.

## Main results claim

At v1.0, on synthetic data only: Learn passed 14 tests with Haar discrete wavelet transform features (12 features, two Haar levels) fed to a hand-written NumPy neural network (hidden 16), reaching test accuracy 1.000 on 100 synthetic test signals (100 of 100, clean 50 of 50 and defect 50 of 50) with final train loss 0.1037; Build passed 20 tests (34 tests in total) on 2,900 synthetic signals (2,400 train and 500 test), with a loader speedup of 3.247x in the committed run (naive two-pass 0.4693 seconds, optimized two-pass 0.1445 seconds, timing varies between runs), train accuracy 0.9838, test accuracy 0.984 (492 of 500), final train loss 0.0712, and a versioned NumPy artifact format that is explicitly not SavedModel, with weights sha256 93e2f8c113d0d7b625caa1089d23afc6d65ae218d829eae986ee3cb7b3808b21 on record; Deep passed 9 tests (43 tests in total), adding a stdlib http.server inference API (GET /health, POST /predict, bad input returns 400) and a CPU edge study where edge is CPU by design and no GPU number is claimed, with float artifact accuracy 0.984 (492 of 500) at 3,530 bytes total (2,896 bytes of weights), an int8 variant labelled a NumPy quantization study and not TFLite at accuracy 0.984, quantized size 1,850 bytes and size ratio 0.6388, latency in-process mean 0.1403 ms over 100 calls against HTTP mean 1.4033 ms over 50 calls (timing varies between runs), a cProfile note labelled not TF Profiler with forward tottime share 0.5949, and TensorFlow unavailable (import fails with ModuleNotFoundError) with no TensorFlow result claimed anywhere.

Every number in that claim comes from a run recorded in the level
READMEs and results files in the repo, on synthetic data only. Read
the limits section below before quoting any of it.

## Results by level

### Level 1, Learn (release v0.1)

A signal arrives as a 1-D sequence of length 64. A hand-written
Haar discrete wavelet transform splits it into approximation and
detail coefficients over two levels, and energy and summary
features are computed from those coefficients: 12 features in all.
A local defect raises detail energy, which a reader can check in a
test. The network is a small NumPy network (hidden 16, ReLU,
softmax cross-entropy) with explicit forward, explicit backward,
and explicit gradient descent steps, and no framework fit call.
Training ran 200 epochs at learning rate 0.05 on 200 train signals.

| Measure | Result |
| --- | --- |
| Test accuracy | 1.000 (100 of 100) |
| Clean correct | 50 of 50 |
| Defect correct | 50 of 50 |
| Train loss, epoch 1 to final | 0.5467 to 0.1037 |
| Tests | 14 |
| Determinism | results_learn.json identical across two runs (verified 2026-11-05) |

### Level 2, Build (release v0.5)

Build scales the Learn method. The pipeline implements the tf.data
ideas as labelled NumPy and stdlib equivalents: parallel map over
chunks with order preserved, an in-memory feature cache, prefetch
style batching, and seeded deterministic batches. The feature
extractor and the network are imported from Learn, not copied.
Training is hand-written, full-batch, 200 epochs, on 2,400 train
signals. The artifact discipline is a versioned NumPy artifact
format: manifest, signatures, and a sha256 over the weight files,
verified on load. It is explicitly not SavedModel.

| Measure | Result |
| --- | --- |
| Dataset | 2,900 synthetic signals (2,400 train, 500 test) |
| Loader speedup, committed run | 3.247x (naive 0.4693 seconds, optimized 0.1445 seconds, two passes; timing varies between runs) |
| Train accuracy | 0.9838 |
| Test accuracy | 0.984 (492 of 500; clean 250 of 250, defect 242 of 250) |
| Train loss, epoch 1 to final | 0.5242 to 0.0712 |
| Artifact | Version 1, NumPy artifact format (not SavedModel), load check passed |
| Weights sha256 | 93e2f8c113d0d7b625caa1089d23afc6d65ae218d829eae986ee3cb7b3808b21 |
| Tests | 20 Build, 34 in total |

### Level 3, Deep (release v1.0)

Deep adds a real-time inference API and an edge deployment study.
The API is stdlib http.server plus NumPy, serving the Build
artifact through its hash-verified loader. It is not TF Serving
and is never called TF Serving. Edge is CPU by design in this
study: no GPU number is claimed, and the assigned BlackBox GPU
node was not used.

| Measure | Result |
| --- | --- |
| Endpoints | GET /health, POST /predict; bad input returns 400 |
| Float artifact, total size | 3,530 bytes (2,896 bytes weight files, plus manifest) |
| Float test accuracy | 0.984 (492 of 500, same synthetic test set as Build, seed 42) |
| int8 variant | NumPy quantization study, not TFLite: 1,850 bytes, size ratio 0.6388, accuracy 0.984 (492 of 500) |
| In-process latency | Mean 0.1403 ms, p95 0.152 ms, over 100 calls (timing varies) |
| HTTP latency | Mean 1.4033 ms, p95 1.7417 ms, over 50 local calls (timing varies) |
| Profiler note | Python cProfile, not TF Profiler: SmallNN.forward tottime 0.008043 s, share 0.5949 of total tottime |
| Tests | 9 Deep, 43 in total |

The judgment page, level-3-deep/JUDGMENT.md, records when
TensorFlow is the right tool for this project and when the NumPy
path is, grounded only in what this repo measured.

## Honest limits

- Synthetic data only, at every level. The signals are sine
  mixtures plus small noise, with a local spike or dip as the
  defect. Accuracy here says nothing about real fabric inspection.
- TensorFlow is unavailable: the import fails with
  ModuleNotFoundError on 2026-11-05. No TensorFlow, TF Serving,
  TFLite, SavedModel, or TF Profiler result exists in this repo.
  Each named item is a labelled substitution or a v1.1 backlog item.
- The loader speedup is one two-pass workload on one machine. A
  second full Build run measured 2.692x. Timing varies between runs.
- The int8 variant still runs in float after dequantization, so no
  int8 compute speedup is claimed. It was measured on one small
  model and one synthetic set.
- Latency is single-machine, single-client, with no load or
  concurrency measurement. No real edge device was measured.
- The NumPy artifact format is local to this repo and is loadable
  by nothing else.
- All runs executed on the Muse VM CPU, not the assigned BlackBox
  GPU node. Edge is CPU by design, as the Deep README states.

## Where to go next

- Tutorial: fanout/tutorial.md in the repo, a step-by-step module
  that runs all three levels and the full test suite.
- Releases: v0.1 (Learn), v0.5 (Build), v1.0 (Learn, Build, Deep)
  on the GitHub repo linked above.
- Book chapter: "The Published Method, Rebuilt" per SPEC.md, held
  as a draft angle in this fan-out set.
- License: code Apache-2.0, tutorials CC BY 4.0.
