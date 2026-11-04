# SPEC: signal-ml-lab (P17)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 3.
Hiring evidence: 3 of 5 (data-leaning roles, record proof) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

signal-ml-lab: EXPAND

Subject: classic ML and deep learning. This project rebuilds his published method in public (wavelet features plus a neural network for fabric defect detection, two AIP Conference Proceedings papers, 2023) [confirmed, career documentation Section C]. Seed repo: drawing-modality-test, the PulseKey system (Section 8).
v1 core [confirmed, v1]: the published method rebuilt clean; a full training and evaluation pipeline on a public dataset; Deep adds a real-time inference API and an edge deployment study.
Added track, TensorFlow depth (register items 226 to 235):
- Learn: one custom TensorFlow training loop, written by hand past Keras fit, on the rebuilt method's data.
- Build: a tf.data pipeline at scale (parallel maps, prefetch, caching) with a speed table against the naive loader; SavedModel discipline: signatures, versioning, load checks.
- Deep: the edge study runs on the TF side too: TensorFlow Serving or TFLite deployment of the model, plus one TF Profiler note that explains a slow training step. Close with the judgment page: when TensorFlow is the right tool, in writing (register item 234).
Fan-out: Book chapter "The Published Method, Rebuilt". Paper angle: the rebuild itself, a public replication of his own published method with open data. Course module: P17-signal-ml-lab.

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
