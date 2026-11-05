# Level 3, Deep: signal-ml-lab

Status: DONE 2026-11-05 (ROADMAP-V3-BATCH). Evidence: code in this folder, 9 unit tests in tests/test_deep_edge.py (43 tests total in the repo suite), measured results below, and results_deep.json written by run_deep.py.

## Honest scope line

Synthetic data only, as in Learn and Build. Deep adds a real-time inference API and an edge deployment study over the shipped Build artifact. No real fabric data and no public dataset are used, and no result here is a real-world fabric result.

Edge is CPU by design in this study. No GPU number is claimed and the assigned BlackBox GPU node was not used. That is a named deduction below. All timing was measured on the Muse VM CPU.

TensorFlow status, verified this run: on 2026-11-05, `python3 -c "import tensorflow"` on the Muse VM returned ModuleNotFoundError: No module named 'tensorflow'. No package was installed and no TensorFlow run was faked. The SPEC Deep TensorFlow items are therefore handled by named substitution, exactly as Learn and Build handled TensorFlow absence:

- TensorFlow Serving: named substitution, not shipped. api.py serves the NumPy Build artifact with the standard library http.server plus NumPy. It is not TF Serving and is never called TF Serving.
- TFLite: named substitution, not shipped. edge.py builds an int8-quantized weight variant in NumPy code (quantize and dequantize the weight arrays). It is labelled a NumPy quantization study, not TFLite, and produces no .tflite file.
- TF Profiler: named substitution, not shipped. The profiler note below uses Python cProfile, labelled a Python profiler, not TF Profiler.
- Judgment page: shipped in full as JUDGMENT.md in this folder, grounded only in what this repo measured.

## What is in the code

- api.py: InferenceService loads the shipped Build artifact (level-2-build/artifacts/model/1) through the Build artifact loader (artifact.load_artifact, imported, not copied, hash verified on load). Features come from the Learn extract_features (imported, not copied), the same Haar wavelet features as Learn and Build. Endpoints: GET /health returns status and artifact version; POST /predict takes a raw signal as a JSON list of 64 numbers, or as {"signal": [64 numbers]}, and returns class, label, score, and both class probabilities. Bad input (invalid JSON, wrong length, non-numeric, non-finite, or a missing signal field) returns a clean 400 JSON error.
- edge.py: edge deployment study on CPU. Model size on disk for the float artifact; an int8-quantized variant (per-array scale, int8 storage, dequantize for inference) with quantized size, accuracy on the same synthetic test set as Build, and latency per call; and an in-process versus HTTP latency comparison (mean and p95 over a stated number of calls).
- run_deep.py: runs the whole study end to end, profiles one inference pass with cProfile, starts the API on a local port (18765) for the HTTP measurements, shuts it down, and writes results_deep.json. The file is deterministic except the named timing fields: every latency value and the profiler timings will vary between runs. A repeat run on 2026-11-05 confirmed every non-timing field identical.
- JUDGMENT.md: the written judgment page (when TensorFlow is the right tool for this project and when the NumPy path is).

## Run it

```text
pip install numpy
python -m unittest discover -s tests -p "test_deep*.py" -v
python level-3-deep/run_deep.py
python -m unittest discover -s tests -v
```

## Measured results (2026-11-05, Muse VM CPU, Python 3.12.3, NumPy 1.26.4)

Test set, stated exactly: the same synthetic test set as Build, generated through the Build data generation path (pipeline.make_large_dataset): 500 signals (250 clean, 250 defect), seed 42. Sample signal for latency: the first signal of that test set.

### Sizes and accuracies

| Variant | Size | Test accuracy (500 synthetic, seed 42) |
| --- | --- | --- |
| Float artifact on disk, total (6 weight files plus manifest.json) | 3,530 bytes | 0.984 (492 of 500) |
| Float artifact, weight files only | 2,896 bytes | (same model) |
| int8 quantized variant, NumPy study (int8 arrays plus one float64 scale per array, comparable array bytes) | 1,850 bytes | 0.984 (492 of 500) |

Per-file float sizes: W1.npy 1,664; W2.npy 384; b1.npy 256; b2.npy 144; mu.npy 224; sd.npy 224; manifest.json 634. The quantized size ratio is 0.6388 of the comparable float array bytes. On this synthetic set the int8 study loses no accuracy; that is a property of this small model and this synthetic set, not a general quantization claim.

### Latencies (measured, will vary: the non-deterministic fields)

| Path | Calls | Mean | p95 | Notes |
| --- | --- | --- | --- | --- |
| In-process predict (feature extraction plus forward) | 100 | 0.1403 ms | 0.152 ms | InferenceService.predict_signal |
| HTTP predict (local server, same machine) | 50 | 1.4033 ms | 1.7417 ms | Mean HTTP overhead ratio about 10.002x the in-process mean |
| int8 model forward only (features precomputed) | 100 | 0.0232 ms | 0.03 ms | Dequantized weights, model step only |
| int8 full call (with feature extraction) | 100 | 0.1215 ms mean | not recorded | Same feature cost as the float path |

Reading of the table, stated plainly: at this model size the HTTP layer costs about ten times the whole in-process computation. Serving overhead, not model arithmetic, is the latency problem at this scale.

### Profiler note (Python cProfile, not TF Profiler)

Workload: one inference pass, per-signal forward over the 500 test feature rows plus one batch predict, profiled with Python cProfile in run_deep.py. Total tottime 0.013521 s. The slowest own-code step found is SmallNN.forward in level-1-learn/signal_ml.py (the network forward pass: matrix products, ReLU, softmax): tottime 0.008043 s over 501 calls, a 0.5949 share (59.49 percent) of total tottime, cumtime 0.013221 s. The next steps are NumPy reduction helpers called from inside forward. Feature extraction is not in this workload (features were precomputed), so this note says the model step dominates the model-only pass, and nothing about the feature step. Profiler timings are measured and will vary between runs.

## Failure modes

- HTTP overhead dominates at this scale: a ThreadingHTTPServer from the standard library is a correctness-first server, not a throughput server. No concurrency, throughput, or load result is claimed; a single client on one machine is all that was measured.
- Bad input is rejected, not repaired: signals of any length other than 64, non-numeric entries, and non-finite values return 400. Real sensor input with a different sampling length would need resampling before this API, which is not implemented.
- Quantization was measured on one small model and one synthetic set: per-array int8 with a single scale per array loses no accuracy here, but per-array scales can fail on weight distributions with outliers, and the dequantized model still runs in float, so no int8 compute speedup is claimed. The latency row for the int8 variant measures a float run of dequantized weights.
- Synthetic separability is not fabric performance: accuracy 0.984 on this set says nothing about real inspection data, as in Learn and Build.
- The NumPy artifact and the quantized variant are local to this repo: neither is loadable by TensorFlow Serving or a TFLite runtime. Calling either a TF deployment would be false.
- Single-machine timing: all latencies are one Muse VM CPU, one run sample, with run to run variance. No edge hardware (no board, no phone, no accelerator) was measured.
- The API has no authentication, no rate limiting, and no TLS: it binds to localhost in the study and is a study server, not a production service.

## Grade: 7,600 of 10,000

Deductions, named and summing to 2,400:
- Synthetic data only, no real fabric data and no public dataset run at Deep: minus 900
- TensorFlow unavailable, so TF Serving, TFLite, and TF Profiler are named substitutions, not shipped: minus 600
- Run on the Muse VM CPU, not the BlackBox GPU fleet node (edge is CPU by design, and no GPU number is claimed): minus 300
- Quantized variant is a NumPy quantization study, not a real TFLite conversion, and its inference still runs in float after dequantization: minus 250
- Single-machine timing with variance, one latency sample per path, no load or concurrency measurement: minus 200
- Edge hardware matrix is one CPU only: no real edge device was measured: minus 150
Total deductions: 2,400. Grade: 7,600 of 10,000.

## Next

The v1.1 backlog is in ROADMAP.md: real TensorFlow and TFLite conversion when TF is available, a real public fabric dataset run, TF Serving deployment, and a larger edge hardware matrix.
