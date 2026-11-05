# DOSSIER: signal-ml-lab (P17) v1, 2026-11-04

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

## Past

Prior art for this project is named in NOVELTY.md (the original published method (Jesun's own AIP papers, 2023)). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 3 of 5 (data-leaning roles, record proof) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

## Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

## Deep ship note (2026-11-05, appended, history kept)

Level 3 Deep DONE 2026-11-05. A real-time inference API (level-3-deep/api.py, stdlib http.server plus NumPy) serves the shipped Build artifact (level-2-build/artifacts/model/1) through the Build artifact loader, with the Learn Haar features imported, a health endpoint, a predict endpoint, and clean 400 handling for bad input. The edge deployment study (level-3-deep/edge.py, CPU only, BlackBox GPU node not used, no GPU number claimed) measured: float artifact 3,530 bytes on disk total (2,896 bytes of weight files); an int8-quantized variant built as a labelled NumPy quantization study (not TFLite) at 1,850 comparable bytes; float and int8 accuracy both 0.984 (492 of 500) on the same synthetic test set as Build (500 signals, seed 42); in-process latency mean 0.1403 ms and p95 0.152 ms over 100 calls; HTTP latency mean 1.4033 ms and p95 1.7417 ms over 50 local calls (timing fields vary between runs). The profiler note uses Python cProfile (labelled a Python profiler, not TF Profiler): the slowest own-code step is SmallNN.forward, tottime 0.008043 s, share 0.5949 of total tottime, over one inference pass over the 500 test signals. The judgment page is level-3-deep/JUDGMENT.md. TensorFlow was verified not importable on 2026-11-05 (ModuleNotFoundError), so TF Serving, TFLite, and TF Profiler are named substitutions, not shipped, and no TensorFlow result is claimed. Grade: 7,600 of 10,000, deductions named in level-3-deep/README.md. The v1.1 backlog (real TensorFlow and TFLite conversion, a real public fabric dataset run, TF Serving deployment, larger edge hardware matrix) is in ROADMAP.md.
