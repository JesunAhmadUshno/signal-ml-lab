# signal-ml-lab

**Question.** Can Haar wavelet features plus a small hand-written
neural network, the method of two 2023 fabric-inspection papers, detect
local defects in 1-D signals, and do the wavelet features beat simpler
inputs?

**Headline.** On 500 synthetic test signals the rebuilt method scores
0.984 (492 of 500, Wilson 95% interval 0.9687 to 0.9919), but it does
not beat simpler inputs: the same network on FFT band energies scores
0.996 (0.9855 to 0.9989) and logistic regression on the same features
0.982. Synthetic data only; the papers' own data and numbers are not
tested.

**Field.** ML and AI research. **Job family.** Applied machine
learning (data-leaning roles).

**Verdict.** Functional, not pre-registered. Tests pass in CI on Python
3.11 and 3.12, one command reproduces each result, and the full system
(features, training, versioned artifact, HTTP inference API, edge
study) ran end to end with its outputs committed. No plan was written
before the results, so the verdict is capped at Functional until a
locked confirmatory cycle (PROPOSAL.md).

Live dashboard: https://jesunahmadushno.github.io/signal-ml-lab/ (runs
the committed model in your browser on the exact seed-42 test set, with
a known-answer check at build time).

## Reproduce

```text
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python level-3-deep/rigor.py            # headline, baselines, intervals -> level-3-deep/results_rigor.json
python level-1-learn/run_learn.py       # Learn run -> level-1-learn/results_learn.json
python level-2-build/run_build.py       # pipeline, training, artifact -> level-2-build/results_build.json
python level-3-deep/run_deep.py         # API and edge study -> level-3-deep/results_deep.json
python level-3-deep/export_site_data.py # dashboard data -> site/src/generated/
```

NumPy only, CPU, seconds. rigor.py, run_learn.py and
export_site_data.py regenerate their files, and CI checks them against
the committed versions with `tests/json_close.py`: structure, strings
and integers (every count and prediction) exactly, floats within 1e-9
relative. Floats are not byte-stable across CPUs: on 2026-11-09 a CI
runner wrote testset.json differing at about 1e-16 relative under the
same pinned NumPy (run 42548252398), so the earlier "byte for byte"
claim held only on the machines tried. run_build.py and run_deep.py also
record timings, which vary between runs; every other field is stable.

## Lifecycle

- Question: DONE 2026-11-04
- Proposal: OPEN (not pre-registered; PROPOSAL.md is a draft written after the results)
- Baseline: DONE 2026-11-07
- Build: DONE 2026-11-05
- Evaluate: OPEN
- Report: DONE 2026-11-08
- Review: OPEN
- Release: OPEN

## Layout

| Path | What it holds |
| --- | --- |
| PROPOSAL.md | Question, prior work, scope and a draft plan for a locked confirmatory cycle (not locked; holds the former SPEC.md, NOVELTY.md and DOSSIER.md) |
| REPORT.md | Background, method, results with intervals, questions scored, misses, threats, future work (holds the former research review, ROADMAP.md backlog and v1.0 benchmark table) |
| level-1-learn/ | Code: Haar transform, 12 features, the NumPy network (signal_ml.py); Learn runner and its results_learn.json |
| level-2-build/ | Code: parallel cached pipeline, training at scale, versioned artifact (artifacts/model/1); results_build.json |
| level-3-deep/ | Code: HTTP inference API, edge study, rigor pass, dashboard export; results_deep.json, results_rigor.json; JUDGMENT.md |
| tests/ | Unit tests and the layout check, run in CI |
| site/ | The live dashboard, built from the results files and the committed weights |
| docs/decisions/ | Records of hard-to-reverse choices |
| fanout/, PUBLICITY.md | Dated 2026-11-05 publicity drafts and the channel log (nothing posted) |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | Change history, citation, security policy, licence |

Evidence and results live next to the code that writes them, in each
level folder, instead of separate experiments/ and results/ folders:
every runner writes one JSON next to itself, and the dashboard
and tests read them at those paths (docs/decisions/0002).

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
