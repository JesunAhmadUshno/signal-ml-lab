# Tutorial: P17 signal-ml-lab, course module

Module: P17-signal-ml-lab
By Jesun Ahmad Ushno
Date: 2026-11-05
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the signal-ml-lab repo, run all three levels, run
the test suite for each level, and read your own results against
the recorded v1.0 numbers. Every command below exists in the repo
as shipped at v1.0 and was run by the author of this module on
2026-11-05 to verify it. The project needs Python 3 with NumPy.
TensorFlow is not needed and is not used: its import fails with
ModuleNotFoundError in the build environment, and no TensorFlow
result is claimed anywhere in this module.

All recorded numbers in this module come from synthetic signals.
They prove the wavelet feature path, the pipeline and artifact
machinery, the serving path, and the edge measurement method. They
do not prove real-world fabric performance.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/signal-ml-lab.git
cd signal-ml-lab
```

Install the one dependency:

```text
pip install numpy
```

## Step 1: Level 1, Learn, wavelet features plus a hand-written network

Read level-1-learn/README.md first. It explains the Haar discrete
wavelet transform over two levels, the 12 features, the synthetic
signal design (length 64, sine mixtures, a local spike or dip as
the defect), and the hand-written NumPy network (hidden 16).

Run the Learn runner:

```text
python level-1-learn/run_learn.py
```

Run the Learn tests:

```text
python -m unittest discover -s tests -p "test_learn*.py" -v
```

Run the scaffold smoke test:

```text
python tests/test_smoke.py
```

Read your results:

- Expected summary: test accuracy 1.000 on 100 synthetic test
  signals (clean 50 of 50, defect 50 of 50), final train loss 0.1037.
- The runner writes level-1-learn/results_learn.json.
- Expected tests: 14 tests, all passing.
- Expected smoke test line: scaffold smoke test: PASS, 8 files present.

## Step 2: Level 2, Build, pipeline at scale and a versioned artifact

Read level-2-build/README.md. The pipeline uses a parallel map,
a feature cache, and prefetch style batching. The artifact is a
versioned NumPy artifact format with a manifest, signatures, and
a sha256 over the weight files. It is explicitly not SavedModel.

Run the Build runner:

```text
python level-2-build/run_build.py
```

Run the Build tests:

```text
python -m unittest discover -s tests -p "test_build*.py" -v
```

Read your results:

- Expected dataset: 2,900 synthetic signals (2,400 train, 500 test).
- Expected training: train accuracy 0.9838, test accuracy 0.984
  (492 of 500), final train loss 0.0712.
- The committed run records a loader speedup of 3.247x (naive
  two-pass 0.4693 seconds, optimized two-pass 0.1445 seconds).
  Timing varies between runs, so your speedup will differ; the
  accuracy and loss fields should match.
- The runner writes level-2-build/results_build.json and checks
  the artifact at level-2-build/artifacts/model/1, weights sha256
  93e2f8c113d0d7b625caa1089d23afc6d65ae218d829eae986ee3cb7b3808b21.
- Expected tests: 20 Build tests, all passing (34 in total with Learn).

## Step 3: Level 3, Deep, inference API and CPU edge study

Read level-3-deep/README.md and level-3-deep/JUDGMENT.md. The API
is stdlib http.server plus NumPy: GET /health and POST /predict,
with bad input returning 400. It is not TF Serving. The int8
variant is a NumPy quantization study, not TFLite. The profiler
note uses Python cProfile, not TF Profiler. Edge is CPU by design,
and no GPU number is claimed.

Run the Deep runner:

```text
python level-3-deep/run_deep.py
```

Run the Deep tests:

```text
python -m unittest discover -s tests -p "test_deep*.py" -v
```

Read your results:

- Expected sizes and accuracies: float artifact 3,530 bytes total
  (2,896 bytes of weights) at accuracy 0.984 (492 of 500); int8
  NumPy study at 1,850 bytes, size ratio 0.6388, accuracy 0.984.
- The committed run records in-process latency mean 0.1403 ms over
  100 calls and HTTP latency mean 1.4033 ms over 50 calls. Timing
  varies between runs, so your latency fields will differ; the
  size and accuracy fields should match.
- The profiler note names SmallNN.forward as the slowest own-code
  step, tottime share 0.5949 in the committed run.
- The runner writes level-3-deep/results_deep.json.
- Expected tests: 9 Deep tests, all passing (43 in total).

## Step 4: Run the full test suite

```text
python -m unittest discover -s tests -v
```

Expected: 43 tests in total (14 Learn, 20 Build, 9 Deep), all
passing, plus the scaffold smoke test.

## Step 5: Check your understanding

1. Why does a local defect raise detail energy? (Answer: the Haar
   detail coefficients measure local change between neighbouring
   samples. A short spike or dip is a sharp local change, so its
   detail coefficients and their energy grow, and a test asserts
   this on a defect signal against a clean one.)
2. Why is the Build artifact never called SavedModel? (Answer:
   SavedModel is a TensorFlow format. This artifact is a set of
   NumPy files with a manifest and a hash check, local to this
   repo. TensorFlow is not importable in this environment, so
   calling it SavedModel would be false.)
3. The int8 study loses no accuracy here. Why is that not a
   general quantization claim? (Answer: it was measured on one
   small model and one synthetic set, with one scale per array,
   and the dequantized model still runs in float. Other weight
   distributions and real data can behave differently.)

## Limits to keep with the numbers

- Synthetic data only, at every level.
- TensorFlow is unavailable and no TensorFlow result is claimed.
- Timing fields (loader speedup, latencies, profiler timings) vary
  between runs. Size, accuracy, and loss fields are the stable ones.
- The int8 variant is a NumPy quantization study, not TFLite, and
  its inference runs in float after dequantization.
