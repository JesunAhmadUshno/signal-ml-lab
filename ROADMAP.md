# ROADMAP: signal-ml-lab

## v1.0 status (2026-11-05)

Learn, Build, and Deep are DONE. Deep shipped 2026-11-05: a real-time inference API (stdlib HTTP plus NumPy, serving the Build artifact), an edge deployment study on CPU (float size 3,530 bytes total, int8 NumPy quantization study at 1,850 bytes, both at 0.984 accuracy on the 500 signal synthetic test set, seed 42), a Python cProfile profiler note (labelled, not TF Profiler), and the judgment page in level-3-deep/JUDGMENT.md. Grade 7,600 of 10,000, deductions in level-3-deep/README.md.

TensorFlow was verified not importable on 2026-11-05 at every level, so the TensorFlow items below were never done silently or claimed: they are the v1.1 backlog.

## v1.1 backlog

- Real TensorFlow and TFLite conversion when TF is available: port the NumPy artifact weights to a real TensorFlow model, convert to TFLite with real int8 quantization, and re-measure size, accuracy, and latency against the NumPy study numbers in level-3-deep.
- TensorFlow custom training loop with GradientTape (no model.fit) on the rebuilt method data, plus a tf.data pipeline (parallel maps, prefetch, caching) measured against the NumPy pipeline speed table in level-2-build.
- SavedModel discipline with the real format: signatures, versioning, load checks, replacing the labelled NumPy artifact format.
- TF Serving deployment of the real model, measured against the stdlib HTTP API latency in level-3-deep, plus a real TF Profiler note on a slow TensorFlow training step.
- A real public fabric dataset run, reported separately from all synthetic results in this repo.
- Larger edge hardware matrix: at least one real edge device or board plus the Muse VM CPU baseline, with the same signals and the same accuracy and latency method as level-3-deep.
- Other wavelets beyond Haar, baseline classifiers, larger synthetic sets, calibration and judge work on the API scores.
