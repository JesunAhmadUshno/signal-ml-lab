# signal-ml-lab

The published wavelet method for fabric defect detection, rebuilt clean in public, with a TensorFlow depth track.

Status: Level 1 Learn DONE and Level 2 Build DONE 2026-11-05 (ROADMAP-V3-BATCH). Deep not started. All measured results below are on synthetic data only.
Project: P17. Fleet node: BlackBox (GPU). Wave: Wave 3.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: DONE 2026-11-05. Haar wavelet features plus a hand-written NumPy neural network on synthetic signals. TensorFlow loop is Build scope (TF not importable at Learn time, see level README).
- Level 2, Build: DONE 2026-11-05. Data pipeline at scale (parallel map, caching, prefetch style batching) with a speed table against a naive loader, hand-written training at larger scale, and versioned NumPy artifact discipline (labelled format, not SavedModel; TensorFlow still not importable, see level README).
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

| Test | Result |
| --- | --- |
| Learn test accuracy (synthetic, 100 test signals) | 1.000 (100 of 100) |
| Learn clean correct | 50 of 50 |
| Learn defect correct | 50 of 50 |
| Learn train loss epoch 1 to final | 0.5467 to 0.1037 |
| Learn unit tests | 14 tests, see tests/test_learn_signal.py |
| Learn grade | 7,300 of 10,000, deductions in level-1-learn/README.md |
| Build dataset (synthetic) | 2,900 signals (2,400 train, 500 test) |
| Build speed, naive two-pass | 0.4693 seconds, Muse VM |
| Build speed, optimized two-pass | 0.1445 seconds, speedup 3.247x |
| Build test accuracy (synthetic) | 0.984 (492 of 500) |
| Build train accuracy | 0.9838 |
| Build train loss epoch 1 to final | 0.5242 to 0.0712 |
| Build unit tests | 20 tests, see tests/test_build_pipeline.py |
| Build artifact | Version 1, NumPy artifact format (not SavedModel), load check passed |
| Build grade | 7,500 of 10,000, deductions in level-2-build/README.md |

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
