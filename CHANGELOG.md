# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 were
cut under the earlier Learn / Build / Deep ladder; under the portfolio
standard v2 the repo is Functional, not a v1.0 release.

## [Unreleased]

### Changed
- 2026-11-08: Restructured to the portfolio standard v2. README is the
  abstract (verdict: Functional, not pre-registered). PROPOSAL.md
  (unlocked draft) holds the former SPEC.md, NOVELTY.md and DOSSIER.md;
  REPORT.md holds the results, the post hoc questions scored with the
  misses, the former docs/research-review.md, the ROADMAP.md backlog
  and docs/index.md. Grades leave the public repo. NumPy pinned. CI
  runs every test file, checks the deterministic results regenerate
  unchanged and has read-only permissions; CodeQL covers the dashboard
  JavaScript; Dependabot covers the dashboard's npm packages. Three
  decision records added.

## [1.0] - 2026-11-05 to 2026-11-07

### Added
- 2026-11-07: Live dashboard (site/) on GitHub Pages, running the
  committed model on the seed-42 test set with a build-time
  known-answer check; export of weights and test set for it.
- 2026-11-07: Rigor pass (level-3-deep/rigor.py): baselines, seed
  variation, Wilson intervals, ROC/PR, miss analysis, difficulty sweep.
  Recorded v1.0 results files unchanged.
- 2026-11-05: Fan-out drafts (fanout/, PUBLICITY.md); nothing posted.
- 2026-11-05: Level 3 Deep: stdlib HTTP inference API and CPU edge
  study (float and int8 sizes, accuracy, latency, cProfile note) and
  the TensorFlow judgment page. Tag v1.0.
- 2026-11-05: CI matrix on Python 3.11 and 3.12 with ruff, CodeQL,
  Dependabot, release, labeler, greetings and stale workflows.
- 2026-11-05: Level 2 Build: parallel cached pipeline with a speed
  table, training at 2,900 synthetic signals, versioned NumPy
  artifact. Tag v0.5.
- 2026-11-05: Level 1 Learn: Haar features plus a hand-written NumPy
  network on synthetic signals. Tag v0.1.

### Removed
- 2026-11-07: Unused stats() helper in extract_features.

## [0.0] - 2026-11-04

### Added
- Wave 0 scaffold: SPEC.md, DOSSIER.md v1, NOVELTY.md draft, level
  folders, CI smoke test, Apache-2.0 licence.
