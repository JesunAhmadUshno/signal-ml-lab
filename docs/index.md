# signal-ml-lab

The published wavelet method for fabric defect detection, rebuilt clean in public: Haar wavelet features plus a hand-written NumPy neural network on synthetic signals. All measured results below are on synthetic data only.

- Learn: Haar wavelet features (12 features, two Haar levels) plus a hand-written NumPy neural network. Test accuracy 1.000 (100 of 100) on the synthetic Learn set.
- Build: a data pipeline at scale (parallel map, caching, prefetch style batching) with a speed table against a naive loader, hand-written training at larger scale, and versioned NumPy artifact discipline. Test accuracy 0.984 (492 of 500) on the synthetic Build set of 2,900 signals; optimized two-pass 0.1445 seconds against naive 0.4693 seconds, speedup 3.247x, measured on the Muse VM.
- Deep: not started.

Grades on record (of 10,000): Learn 7,300, Build 7,500.

TensorFlow status, honest: TensorFlow is not available in the build environment and is not claimed. The TensorFlow items in the spec are implemented as labelled NumPy and standard library equivalents, with a named grade deduction at each level. The Build artifact is a labelled NumPy artifact format, not SavedModel.

Every number above is already published in this repo (README.md and the level write-ups) and reproduced by the runners in this repo, checked in CI. The full write-ups are in the level folders: level-1-learn, level-2-build, level-3-deep.

Releases: https://github.com/JesunAhmadUshno/signal-ml-lab/releases
