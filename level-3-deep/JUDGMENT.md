# Judgment: when TensorFlow is the right tool for this project

Written for Level 3 Deep (register item 234 in SPEC). Grounded only in
what this repo actually measured. No claim here rests on a TensorFlow
run, because no TensorFlow run exists in this repo.

## The measured facts this judgment rests on

- The model is tiny by construction: 12 Haar features, one hidden layer
  of 16 units, 2 classes. The float artifact is 3,530 bytes on disk
  total (2,896 bytes of weight files). In-process prediction on the
  Muse VM CPU takes a mean of 0.1403 ms per call including feature
  extraction (100 calls, timing varies).
- The NumPy int8 quantization study shrinks the comparable array bytes
  from 2,896 to 1,850 (ratio 0.6388) with no accuracy change on the
  synthetic test set: 0.984 (492 of 500) for both float and int8.
- The HTTP layer, not the model, dominates served latency: HTTP mean
  1.4033 ms versus in-process mean 0.1403 ms, about 10.0x overhead on
  one local machine (50 HTTP calls, timing varies).
- The Python profiler (cProfile, not TF Profiler) finds the network
  forward pass as the slowest own-code step: SmallNN.forward took
  0.008043 s of 0.013521 s total tottime (share 0.5949) over one
  inference pass over the 500 test signals.
- TensorFlow was verified not importable on 2026-11-05
  (ModuleNotFoundError). Nothing in Learn, Build, or Deep uses it.

## When the NumPy path is the right tool here

- At this model size, now: a 3.5 KB artifact and sub-millisecond
  in-process inference leave nothing for a serving framework to speed
  up. The standard library API adds no dependency, loads through the
  hash-verified Build artifact loader, and its behaviour is fully
  visible in about 150 lines.
- For teaching and audit: every step (Haar features, forward pass,
  quantization, dequantization) is readable NumPy with no framework
  between the reader and the arithmetic. That is the point of a public
  rebuild of a published method.
- For the edge study as scoped: CPU-only, single signal at a time,
  no batch throughput target. Quantization here is a measurement
  exercise, and the NumPy study measures the real effect (size down,
  accuracy unchanged on this synthetic set) without claiming to be a
  deployment format.

## When TensorFlow is the right tool for this project

- Real fabric data at real scale: if the public-dataset run in the
  v1.1 backlog brings orders of magnitude more signals, longer signals,
  or image inputs, hand-written full-batch NumPy training and a
  per-signal Python feature loop become the bottleneck the Build speed
  table already hints at. tf.data, GPU training, and a framework
  training loop then earn their complexity.
- A real deployment target: TensorFlow Serving or a TFLite runtime is
  the right tool when the model must run on actual edge hardware, be
  served to other systems with a standard model format, or be updated
  and versioned by a team. The NumPy artifact format in this repo is
  local to this repo and loadable by nothing else, which the Build
  README already states as a failure mode.
- Profiling a real training step: TF Profiler answers questions about
  a TensorFlow training step. This repo has no TensorFlow training
  step, so its profiler note is honestly a Python cProfile note.
  Claiming a TF Profiler finding here would be false.
- Larger architectures: anything beyond a 12 feature, 16 hidden unit
  network (convolutions over raw signals, sequence models, learned
  features instead of Haar features) should not be hand-written in
  NumPy for production use. Autodiff, tested kernels, and hardware
  coverage are exactly what the framework provides.

## Verdict for this project at v1.0

Keep the NumPy path as the shipped v1.0 path: it is measured, tested,
and sufficient for the model as built. Treat TensorFlow as the v1.1
migration triggered by any one of: real data, a real edge or serving
target, or a larger architecture. That migration is named in
ROADMAP.md, with the conversion, TFLite, and TF Serving steps listed
as backlog, not as done work.
