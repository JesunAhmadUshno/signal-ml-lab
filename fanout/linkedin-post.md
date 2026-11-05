# LinkedIn post draft: P17 signal-ml-lab v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-05
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in the
post body. The first comment line is at the end of this file.

## Post body

I published a wavelet method for fabric defect detection in 2023.
This month I rebuilt it clean, in public, and measured every step.

signal-ml-lab v1.0 is done: Learn, Build, and Deep, on synthetic
data only, labelled synthetic at every level. TensorFlow is not
importable in the build environment, so nothing here claims a
TensorFlow result.

Learn: Haar wavelet features (12 features, two levels) plus a
hand-written NumPy network. Test accuracy 1.000 on 100 synthetic
test signals, final train loss 0.1037. 14 tests. Grade 7,300 of
10,000.

Build: a pipeline at scale (parallel map, caching, prefetch style
batching) on 2,900 synthetic signals. Loader speedup 3.247x in the
committed run, and timing varies between runs. Train accuracy
0.9838, test accuracy 0.984 (492 of 500), final train loss 0.0712.
The artifact is a versioned NumPy format, explicitly not
SavedModel, with its weights sha256 on record. 34 tests in total.
Grade 7,500 of 10,000.

Deep: a stdlib http.server inference API (GET /health, POST
/predict, bad input returns 400) and a CPU edge study. Edge is CPU
by design, no GPU number claimed. Float artifact: 3,530 bytes
total, accuracy 0.984. The int8 variant is a NumPy quantization
study, not TFLite: 1,850 bytes, ratio 0.6388, accuracy 0.984.
In-process latency mean 0.1403 ms over 100 calls, HTTP mean
1.4033 ms over 50 calls, timing varies. The profiler note is
Python cProfile, labelled not TF Profiler, with forward tottime
share 0.5949. 43 tests in total. Grade 7,600 of 10,000, with
deductions named in the repo.

The honest limits, in the post and not in a footnote: synthetic
signals only, so these accuracies say nothing about real fabric;
every TensorFlow item (TF Serving, TFLite, SavedModel, TF
Profiler) is a named substitution or v1.1 backlog, not a shipped
result; and the int8 model still runs in float after
dequantization, so no int8 speedup is claimed.

If you rebuild published work: do you label your substitutions
this plainly, or does the framework name end up in the headline
anyway?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/signal-ml-lab
