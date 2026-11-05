# signal-ml-lab

The published wavelet method for fabric defect detection, rebuilt clean in public, with a TensorFlow depth track.

Status: Level 1 Learn DONE 2026-11-05 (ROADMAP-V3-BATCH). Build and Deep not started. Measured Learn results below are on synthetic data only.
Project: P17. Fleet node: BlackBox (GPU). Wave: Wave 3.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: DONE 2026-11-05. Haar wavelet features plus a hand-written NumPy neural network on synthetic signals. TensorFlow loop is Build scope (TF not importable at Learn time, see level README).
- Level 2, Build: a real working system other people can run.
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

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
