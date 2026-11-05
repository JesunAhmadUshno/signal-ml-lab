# Website project page copy: signal-ml-lab

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-05.
Nothing on any website was changed by this file.

## Headline

signal-ml-lab: the published wavelet method for fabric defect detection, rebuilt clean in public

## Summary

signal-ml-lab is an open machine learning project built by Jesun
Ahmad Ushno as project P17 in Roadmap v3. It rebuilds, in public,
the method from his own AIP Conference Proceedings papers (2023):
Haar wavelet features plus a neural network for fabric defect
detection. It builds the method in three levels: wavelet features
and a hand-written NumPy network, a data pipeline and training at
larger scale with a versioned artifact, and a stdlib HTTP
inference API with a CPU edge deployment study. All data in v1.0
is synthetic and labelled synthetic. TensorFlow is unavailable in
the build environment, and no TensorFlow result is claimed.

Status: v1.0, 2026-11-05. Learn, Build, and Deep are done. Total
tests: 43, all passing at v1.0. Level grades: Learn 7,300 of
10,000, Build 7,500 of 10,000, Deep 7,600 of 10,000.

## Results

All results below are measured runs on synthetic data, recorded
in the level READMEs in the GitHub repo. They prove the feature
path, the pipeline and artifact machinery, and the edge
measurement method, not real-world fabric performance.

| Level | Set | Headline result | Notes |
| --- | --- | --- | --- |
| Learn | 300 synthetic signals (200 train, 100 test) | Test accuracy 1.000 (100 of 100), final train loss 0.1037 | Haar DWT features, 12 features, hand-written NumPy network, hidden 16; 14 tests |
| Build | 2,900 synthetic signals (2,400 train, 500 test) | Test accuracy 0.984 (492 of 500), train accuracy 0.9838, loader speedup 3.247x in the committed run (timing varies) | Versioned NumPy artifact, explicitly not SavedModel, weights sha256 on record; 20 tests, 34 total |
| Deep | Same 500 synthetic test signals, seed 42 | Float and int8 accuracy both 0.984 (492 of 500) | API GET /health, POST /predict, bad input 400; float 3,530 bytes total, int8 NumPy study (not TFLite) 1,850 bytes, ratio 0.6388; 9 tests, 43 total |

## Honest limits

- Synthetic data only, at every level, written as a teaching
  stand-in for fabric inspection signals.
- TensorFlow is unavailable (import fails with
  ModuleNotFoundError). TF Serving, TFLite, SavedModel, and TF
  Profiler are named substitutions or v1.1 backlog, never claimed.
- Edge is CPU by design. No GPU number is claimed, and no real
  edge device was measured.
- Timing values vary between runs. The Deep cProfile note is a
  Python profiler note, labelled not TF Profiler.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/signal-ml-lab
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v0.5
- Release v1.0 (Learn, Build, Deep): https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
