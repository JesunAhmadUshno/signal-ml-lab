# Research review: signal-ml-lab

Graded 2026-11-07 against a doctoral-level research rubric and an engineering rubric. Every number below comes from `level-1-learn/results_learn.json`, `level-2-build/results_build.json`, `level-3-deep/results_deep.json` or `level-3-deep/results_rigor.json`.

**Overall: about 5.5/10 as research (4.5/10 before the rigor pass), about 7.5/10 as engineering.** The code is clean, tested and reproducible, and its limits are disclosed. The scientific claim stays thin because the task is synthetic and close to separable.

## Scorecard

| Criterion | Grade | Evidence and verdict |
| --- | --- | --- |
| Research question | 5/10 | Implicit: can Haar wavelet features plus a small network detect local defects in 1-D signals? The stated goal (rebuild the 2023 fabric-inspection method) is not tested against the paper's data or numbers. |
| Hypotheses | 4/10 | None written down before v1.0. The rigor pass tests one implicitly: wavelet features beat other inputs. The answer on this task is no (FFT band energies 0.996 vs wavelets 0.984). |
| Method | 6/10 | Clear and fully specified in code: 12 features from two Haar levels, a 12x16x2 ReLU/softmax network trained by full-batch gradient descent. |
| Baselines | 7/10 (was 2) | Fixed now: majority 0.500, one-feature threshold 0.934, logistic regression 0.982, same network on raw samples 0.980, on FFT energies 0.996. |
| Statistics and uncertainty | 7/10 (was 2) | Fixed now: Wilson 95% intervals on every accuracy (headline 0.9687 to 0.9919), 10 training seeds (mean 0.9832, sd 0.0018), 5 fresh data seeds (mean 0.9752), ROC AUC 0.9938, average precision 0.9955. No paired test between models (all intervals overlap except majority and threshold). |
| Reproducibility | 9/10 | Seeded, NumPy only, CI on Python 3.11 and 3.12, hash-verified artifact. The dashboard recomputes 492/500 and the int8 result in the browser and the build fails if it disagrees with the Python predictions. |
| Threats to validity | 6/10 | Synthetic data, shared generator for train and test, one CPU machine, single-run latencies. Now measured: the model fails under unseen noise (false alarms 0.705 at sigma 0.1, 1.0 from sigma 0.2) and on width-4 bumps aligned to a level-2 Haar block (7 of the 8 test misses, although only 12 of 250 test defects are aligned). |
| Related work | 2/10 | NOVELTY.md is a placeholder and DOSSIER.md is short. No survey and no comparison with published numbers. |
| Limitations | 8/10 | README, JUDGMENT.md and the dashboard state the TensorFlow substitutions, synthetic data and CPU-only scope. |
| Engineering rigor | 8/10 | 50 unit tests, a versioned artifact whose loader rejects tampering, a parallel cached pipeline with an equality check, a stdlib HTTP API. Latency has no warm-up or repeat runs. |

## What the rigor pass changed (cheap gaps fixed now)

- `level-3-deep/rigor.py` (CPU, seconds) adds baselines, seeds, Wilson intervals, ROC/PR, miss analysis and a difficulty sweep, written to the new `results_rigor.json`. The recorded v1.0 files are unchanged.
- Training budget: the v1.0 headline measures an under-trained network. lr 0.05 for 2,000 epochs gives 0.994; lr 0.2 for 2,000 epochs gives 1.000 (Wilson 0.9924 to 1.0).
- Difficulty sweep at training noise: detection is 0.515 at amplitude 0.5, 0.895 at 0.8 and 0.975 at 1.4.
- `level-3-deep/export_site_data.py` exports the committed weights and the seed-42 test set so the dashboard can do live inference with a build-time parity check.
- Removed the dead `stats()` helper in `extract_features`; corrected the README tagline and the stale Deep line in `docs/index.md`.

## Remaining gaps and exactly what each needs

1. **Validation on real data.** Needs a public 1-D or image fabric-defect dataset (for example AITEX or TILDA, which are images, so rows or texture profiles would have to be extracted into 1-D signals) and a written extraction protocol. CPU is enough at this model size; minutes per run.
2. **Comparison with the 2023 papers.** Needs full citations for the two 2023 AIP Conference Proceedings papers (SPEC.md), their datasets and their reported metrics, then the same metrics computed here. No compute beyond the item above.
3. **Related-work survey.** Needs a literature pass (wavelet and FFT features for defect detection, 1-D CNN baselines) of about 10 to 20 papers, written into DOSSIER.md and NOVELTY.md. No compute.
4. **Distribution shift.** Needs a held-out generator variant (different base frequencies, drift, non-Gaussian noise) and training with noise augmentation over sigma 0.05 to 0.3. CPU, under a minute with rigor.py's generator.
5. **Fix for the alignment blind spot.** Needs a shift-invariant transform (undecimated or stationary Haar transform, or features at both even and odd offsets) and a re-run of the width-4 start sweep. CPU, seconds.
6. **Paired model comparison.** Needs McNemar tests on the shared test set or a bootstrap over several fresh test sets. CPU, seconds; can be added to rigor.py.
7. **Latency statistics.** Needs warm-up plus repeated runs (for example 30 runs of 1,000 calls) with medians and intervals, ideally on a second machine. CPU, a few minutes.
8. **TensorFlow track.** Needs an environment where TensorFlow imports, then SavedModel, TF Serving, TFLite conversion and the TF Profiler. A real edge result also needs real edge hardware (for example a Raspberry Pi or a microcontroller board).
