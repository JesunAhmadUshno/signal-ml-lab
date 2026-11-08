# PROPOSAL: signal-ml-lab (P17)

Status: DRAFT, NOT LOCKED, NOT PRE-REGISTERED. Written 2026-11-08, after
every result in this repo existed. Git history shows no plan with
hypotheses and pass lines committed before the first run (the Wave 0
scaffold of 2026-11-04 carried scope and acceptance criteria only), so
nothing here may be read as a pre-registration. The repo is capped at
Functional until a confirmatory cycle is locked under a tag and run
(docs/decisions/0003).

## Question

Can Haar wavelet features plus a small hand-written neural network, the
method of two 2023 fabric-inspection papers (AIP Conference
Proceedings), detect local defects in 1-D signals, and do the wavelet
features beat simpler inputs?

## Questions the exploratory work answered (post hoc, not locked)

These were asked after the runs; REPORT.md scores them, misses
included. They are exploratory, not confirmatory.

- Q1. The rebuilt method separates clean from defect signals on the
  repo's synthetic recipe (accuracy well above the 0.5 majority rate).
- Q2. Wavelet features beat simpler inputs under the same network and
  budget (raw samples, FFT band energies) and simpler models (one
  threshold, logistic regression).
- Q3. A per-array int8 quantization of the weights loses no accuracy.
- Q4. At this model size, HTTP serving overhead, not model arithmetic,
  dominates served latency.

## Draft plan for a locked confirmatory cycle (not yet locked)

To lift the cap, a later pull request would fix, before any run:
data (a public fabric-defect dataset with a written 1-D extraction
protocol, or the 2023 papers' data and metrics if citable), the
hypotheses with numeric pass lines, the baselines and ablations under
one budget (REPORT.md, Remaining gaps 1 to 6), the analysis (Wilson
intervals, McNemar paired tests), and then tag that commit. That
choice of data and venue is the Owner's.

---

The sections below are the original planning files, moved here
verbatim on 2026-11-08 so the history stays in one place.

## Former SPEC.md (Wave 0 scaffold, 2026-11-04)


Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 3.
Hiring evidence: 3 of 5 (data-leaning roles, record proof) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

signal-ml-lab: EXPAND

Subject: classic ML and deep learning. This project rebuilds his published method in public (wavelet features plus a neural network for fabric defect detection, two AIP Conference Proceedings papers, 2023) [confirmed, career documentation Section C]. Seed repo: drawing-modality-test, the PulseKey system (Section 8).
v1 core [confirmed, v1]: the published method rebuilt clean; a full training and evaluation pipeline on a public dataset; Deep adds a real-time inference API and an edge deployment study.
Added track, TensorFlow depth (register items 226 to 235):
- Learn: one custom TensorFlow training loop, written by hand past Keras fit, on the rebuilt method's data.
- Build: a tf.data pipeline at scale (parallel maps, prefetch, caching) with a speed table against the naive loader; SavedModel discipline: signatures, versioning, load checks.
- Deep: the edge study runs on the TF side too: TensorFlow Serving or TFLite deployment of the model, plus one TF Profiler note that explains a slow training step. Close with the judgment page: when TensorFlow is the right tool, in writing (register item 234).
Fan-out: Book chapter "The Published Method, Rebuilt". Paper angle: the rebuild itself, a public replication of his own published method with open data. Course module: P17-signal-ml-lab.

### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

## Former NOVELTY.md (2026-11-04)


Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

### Proposed claim

Against the original published method (Jesun's own AIP papers, 2023): this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| the original published method (Jesun's own AIP papers, 2023) | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

## Former DOSSIER.md (v1 2026-11-04, Deep note 2026-11-05)


Written under the documentation and publishing standard (past, present, future). Updated at ship time.

### Past

Prior art for this project is named in NOVELTY.md (the original published method (Jesun's own AIP papers, 2023)). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

### Present

Current demand evidence: hiring value 3 of 5 (data-leaning roles, record proof) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

### Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

### Deep ship note (2026-11-05, appended, history kept)

Level 3 Deep DONE 2026-11-05. A real-time inference API (level-3-deep/api.py, stdlib http.server plus NumPy) serves the shipped Build artifact (level-2-build/artifacts/model/1) through the Build artifact loader, with the Learn Haar features imported, a health endpoint, a predict endpoint, and clean 400 handling for bad input. The edge deployment study (level-3-deep/edge.py, CPU only, BlackBox GPU node not used, no GPU number claimed) measured: float artifact 3,530 bytes on disk total (2,896 bytes of weight files); an int8-quantized variant built as a labelled NumPy quantization study (not TFLite) at 1,850 comparable bytes; float and int8 accuracy both 0.984 (492 of 500) on the same synthetic test set as Build (500 signals, seed 42); in-process latency mean 0.1403 ms and p95 0.152 ms over 100 calls; HTTP latency mean 1.4033 ms and p95 1.7417 ms over 50 local calls (timing fields vary between runs). The profiler note uses Python cProfile (labelled a Python profiler, not TF Profiler): the slowest own-code step is SmallNN.forward, tottime 0.008043 s, share 0.5949 of total tottime, over one inference pass over the 500 test signals. The judgment page is level-3-deep/JUDGMENT.md. TensorFlow was verified not importable on 2026-11-05 (ModuleNotFoundError), so TF Serving, TFLite, and TF Profiler are named substitutions, not shipped, and no TensorFlow result is claimed. The v1.1 backlog (real TensorFlow and TFLite conversion, a real public fabric dataset run, TF Serving deployment, larger edge hardware matrix) is in ROADMAP.md.
