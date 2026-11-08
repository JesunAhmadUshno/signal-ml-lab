# 0001: Ship NumPy and standard-library substitutes for the TensorFlow track

Date: 2026-11-05 (recorded 2026-11-08). Status: accepted. Decided by: Jesun Ahmad Ushno.

## Context

The scope (PROPOSAL.md, former SPEC.md) asked for a TensorFlow depth
track: a custom training loop, tf.data, SavedModel, TF Serving or
TFLite, and a TF Profiler note. On 2026-11-05 `import tensorflow` failed
with ModuleNotFoundError on the build machine (torch was absent too),
and no large package was to be installed silently.

## Decision

Build every item in NumPy or the standard library and label it as a
substitute everywhere: a hand-written training loop, a parallel cached
pipeline, a hash-verified NumPy artifact (never called SavedModel), a
stdlib HTTP API (never called TF Serving), a NumPy int8 study (never
called TFLite), and a cProfile note (never called TF Profiler). No
TensorFlow result is claimed.

## Consequences

The model, artifact and API depend on NumPy only and are fully
readable. The artifact is loadable by nothing outside this repo. The
TensorFlow track stays open as future work (REPORT.md, Remaining gaps
8); level-3-deep/JUDGMENT.md says when TensorFlow becomes the right
tool.
