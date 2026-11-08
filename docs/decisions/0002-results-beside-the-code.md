# 0002: Keep results files beside the code that writes them

Date: 2026-11-08. Status: accepted. Decided by: Jesun Ahmad Ushno.

## Context

The portfolio standard v2 layout has experiments/ and results/
folders. Here each runner (level-1-learn/run_learn.py,
level-2-build/run_build.py, level-3-deep/run_deep.py,
level-3-deep/rigor.py) writes one JSON next to itself, and the
dashboard, the tests and the published fan-out pieces all read those
paths.

## Decision

Keep the level folders as domain-named code folders and keep each
results file where its runner writes it. The README layout table maps
them. The inputs are seeded generators, not captured raw data, so there
is no separate raw-evidence folder to keep.

## Consequences

No path breaks for the dashboard, tests or dated documents. CI proves
the deterministic files regenerate unchanged. A future confirmatory
cycle on real data would add experiments/ for that data.
