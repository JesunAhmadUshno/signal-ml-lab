# signal-ml-lab

The published wavelet method for fabric defect detection, rebuilt clean in public. A TensorFlow depth track was planned, but TensorFlow was never importable, so its items shipped as labelled NumPy and standard-library substitutions.

Live dashboard: https://jesunahmadushno.github.io/signal-ml-lab/ (runs the committed model in your browser on the exact seed-42 test set, with a known-answer check at build time).

Status: Level 1 Learn DONE, Level 2 Build DONE, and Level 3 Deep DONE 2026-11-05 (ROADMAP-V3-BATCH). Release v1.0. All measured results below are on synthetic data only.
Project: P17. Fleet node: BlackBox (GPU). Wave: Wave 3.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: DONE 2026-11-05. Haar wavelet features plus a hand-written NumPy neural network on synthetic signals. TensorFlow loop is Build scope (TF not importable at Learn time, see level README).
- Level 2, Build: DONE 2026-11-05. Data pipeline at scale (parallel map, caching, prefetch style batching) with a speed table against a naive loader, hand-written training at larger scale, and versioned NumPy artifact discipline (labelled format, not SavedModel; TensorFlow still not importable, see level README).
- Level 3, Deep: DONE 2026-11-05. Real-time inference API (stdlib HTTP plus NumPy, serving the Build artifact through its loader) and an edge deployment study on CPU (float and int8 sizes, accuracies, latencies, Python cProfile note labelled not TF Profiler, judgment page in level-3-deep/JUDGMENT.md). TensorFlow still not importable, so TF Serving, TFLite, and TF Profiler are named substitutions, not shipped (see level README).

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
| Deep float artifact size on disk | 3,530 bytes total (2,896 bytes weight files) |
| Deep int8 quantized size (NumPy study, not TFLite) | 1,850 bytes (ratio 0.6388 of float array bytes) |
| Deep float test accuracy (synthetic, 500 test signals, seed 42) | 0.984 (492 of 500) |
| Deep int8 test accuracy (synthetic, same set) | 0.984 (492 of 500) |
| Deep in-process latency (100 calls, Muse VM CPU) | mean 0.1403 ms, p95 0.152 ms (timing varies) |
| Deep HTTP latency (50 calls, local, Muse VM CPU) | mean 1.4033 ms, p95 1.7417 ms (timing varies) |
| Deep profiler slowest step (cProfile, not TF Profiler) | SmallNN.forward, 0.008043 s tottime, share 0.5949 |
| Deep unit tests | 9 tests, see tests/test_deep_edge.py (43 tests total) |
| Deep grade | 7,600 of 10,000, deductions in level-3-deep/README.md |

## Rigor pass (2026-11-07)

level-3-deep/rigor.py adds baselines, seed variation, Wilson 95% intervals, ROC/PR, a miss analysis and a difficulty sweep, written to level-3-deep/results_rigor.json (the v1.0 results files are unchanged). Headline: 0.984 (Wilson 95% 0.9687 to 0.9919); one threshold on one feature gets 0.934; the same network on FFT band energies gets 0.996; training longer (lr 0.2, 2,000 epochs) reaches 1.000. The research grade and remaining gaps are in docs/research-review.md.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## Fan-out

Written fan-out set for v1.0, dated 2026-11-05, in fanout/:

- fanout/blog-post.md: "The Published Method, Rebuilt", the full
  story with the main results claim gated by Jev claim_check
  (model jev-1.13.0, noul 0.86, route ACT, no block).
- fanout/tutorial.md: course module P17-signal-ml-lab. Every
  command in it was executed and passed on 2026-11-05 (all three
  runners, all per-level tests, full suite at 43 tests).
- fanout/website-page.md: project page copy only, not published.
- fanout/linkedin-post.md: DRAFT NOT POSTED, link as the
  first-comment line per the LinkedIn rule.
- fanout/paper-candidate.md: public replication brief at
  CANDIDATE. Submission is Jesun's tap, nothing submitted.
- fanout/release-notes.md: draft notes for the live releases
  v0.1, v0.5, and v1.0.

Publicity grades and the channel log are in PUBLICITY.md. Overall
fan-out grade: 9,530 of 10,000. The loop never posts to social:
all social channels are held with named reasons in PUBLICITY.md.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
