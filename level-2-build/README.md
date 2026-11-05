# Level 2, Build: signal-ml-lab

Status: DONE 2026-11-05 (ROADMAP-V3-BATCH). Evidence: code in this folder, 20 unit tests in tests/test_build_pipeline.py, measured results below, and a versioned artifact under artifacts/model/1/. Deep is not started.

## Honest scope line

Synthetic data only. This level scales the Level 1 Learn method to 2,900 synthetic signals with a data pipeline, a hand-written training loop, and versioned model artifact discipline. No real fabric data and no public dataset are used at this level, and no result here is a real-world fabric result.

TensorFlow status, verified this run: on 2026-11-05, `python3 -c "import tensorflow"` on the Muse VM returned ModuleNotFoundError: No module named 'tensorflow'. `python3 -c "import torch"` also failed with ModuleNotFoundError. NumPy 1.26.4 was present on Python 3.12.3. No large package was installed and no TensorFlow run was faked. The SPEC Build items are therefore implemented as labelled NumPy and standard library equivalents, and that substitution carries a named deduction below, as in Learn:

- tf.data parallel map, caching, and prefetch: implemented in pipeline.py with ProcessPoolExecutor over chunks, an in-memory feature cache, a prefetch style iterator, and seeded deterministic batching.
- Custom training loop past framework fit: implemented in train_build.py with explicit forward, backward, and gradient descent steps through the Learn SmallNN class. No framework fit call is used.
- SavedModel discipline: implemented in artifact.py as a labelled NumPy artifact format (signatures, versioning, hash verified load checks). It is NOT SavedModel and is never called SavedModel.

The feature extractor and the network are imported from level-1-learn/signal_ml.py, not copied, so Learn and Build cannot diverge.

## What is in the code

- pipeline.py: naive_features (per-signal loop, no caching), OptimizedPipeline (parallel map over chunks with order preserved, feature caching, deterministic batch_indices and batches), PrefetchBatches (prepares the next batch while the current one is consumed), make_large_dataset (seeded synthetic, labelled synthetic).
- train_build.py: train_at_scale, full-batch hand-written training on pipelined features, with standardisation from train statistics, a loss curve, and train and test accuracy.
- artifact.py: save_artifact and load_artifact for the versioned NumPy artifact format. The manifest per version records the input signature (feature dim 12), the output signature (2 classes, clean and defect), the weight file list, a sha256 over the weight files, and the training config. Loading verifies the format, version, signatures, file list, and hash before constructing the model, and rejects a tampered manifest or weights file.
- run_build.py: runs both loaders on the timing workload, trains, saves and load-checks the artifact, verifies loaded predictions match the trained model, and writes results_build.json.

## Run it

```text
pip install numpy
python -m unittest discover -s tests -p "test_build*.py" -v
python -m unittest discover -s tests -p "test_learn*.py" -v
python level-2-build/run_build.py
python tests/test_smoke.py
```

## Measured results (2026-11-05, Muse VM, Python 3.12.3, NumPy 1.26.4)

Dataset: 2,400 train signals (1,200 clean, 1,200 defect) and 500 test signals (250 clean, 250 defect), 2,900 total, all synthetic and seeded. Feature dim 12. Batches: 19 batches of 128 (last batch smaller), batch seed 123, covering all 2,400 train rows exactly once.

### Speed table, measured

Timing workload, stated exactly: two full passes over the 2,400 training signals. The naive loader extracts features on both passes. The optimized pipeline extracts once in parallel (4 workers, chunk size 256) and serves the second pass from cache, which is the caching benefit this level demonstrates. Both loaders produce identical feature arrays, asserted in tests and in the run.

| Loader | Two-pass seconds | Notes |
| --- | --- | --- |
| Naive loader | 0.4693 | Plain per-signal loop, extraction on both passes |
| Optimized pipeline | 0.1445 | Parallel first pass, cached second pass |
| Speedup | 3.247x | Ratio of the two rows above |

Run to run variance, named: a second full run measured naive 0.4645 seconds, optimized 0.1726 seconds, speedup 2.692x. Timing values are the only fields that differ between the two runs of results_build.json; every non-timing field, including the artifact weights sha256, was identical across both runs (verified by diff on 2026-11-05). Timings are measured on the Muse VM, not on the BlackBox GPU fleet node.

Pipeline counters from the committed run: extraction count 2,400 (one pass only), cache misses 1, cache hits 2 (the second timing pass and the batching pass).

### Training results, measured

Hand-written network, hidden 16, 200 epochs, learning rate 0.05, seed 1, full-batch gradient descent on the pipelined features.

| Metric | Measured |
| --- | --- |
| Train accuracy | 0.9838 |
| Test accuracy | 0.984 (492 of 500) |
| Clean correct | 250 of 250 |
| Defect correct | 242 of 250 |
| Train loss epoch 1 | 0.5242 |
| Train loss epoch 50 | 0.1267 |
| Train loss epoch 100 | 0.0928 |
| Train loss epoch 150 | 0.0791 |
| Train loss epoch 200 (curve) | 0.0714 |
| Train final loss | 0.0712 |

Note, stated plainly: at this larger scale the synthetic test accuracy is 0.984, not the 1.000 seen on the small Learn set. The larger seeded set is slightly harder, and the defect class carries all 8 test errors.

### Artifact evidence

- Layout: artifacts/model/1/ with manifest.json and weight files W1.npy, b1.npy, W2.npy, b2.npy, mu.npy, sd.npy.
- Format: signal-ml-lab numpy artifact v1 (not SavedModel).
- Input signature: feature dim 12, float64. Output signature: 2 classes (clean, defect).
- Weights sha256: 93e2f8c113d0d7b625caa1089d23afc6d65ae218d829eae986ee3cb7b3808b21 (over the weight files in manifest order).
- Load check in the run: passed. Loaded artifact predictions match the trained model exactly, and the loaded artifact test accuracy is 0.984.
- Tamper handling, tested: a flipped byte in a weights file, a replaced hash in the manifest, and a changed feature dim in the manifest each make load_artifact raise ValueError.

## Failure modes

- Cache benefit depends on reuse: a single pass with no second use pays the parallel extraction cost and gains little. The speed table is a two-pass workload and says so.
- Parallel map uses processes: on a single-core machine, or with tiny datasets, process startup can erase the gain. The pipeline falls back to in-process extraction when workers is 1 or there is one chunk.
- Synthetic separability is not fabric performance: the anomaly in these signals is a local spike or dip by construction, and accuracy here says nothing about real inspection data.
- Artifact format is local to this repo: the NumPy artifact is not loadable by TensorFlow Serving or TFLite, and calling it SavedModel would be false. Porting it needs TensorFlow, which is not available here.
- Full-batch training memory grows with dataset size: features for 2,400 signals are small, but this loop form does not stream, so much larger sets need the batched path used for gradient steps.
- Hash verification detects tampering and corruption, not a maliciously rebuilt manifest plus weights pair: a re-signed fake artifact would pass the hash check, because there is no key or signature authority in this format.

## Grade: 7,500 of 10,000

Deductions, named and summing to 2,500:
- Synthetic signals only, no public dataset and no real fabric data at Build: minus 1,000
- TensorFlow not available, so tf.data, the GradientTape loop, and SavedModel are labelled NumPy and stdlib equivalents instead of the real framework: minus 700
- Run on the Muse VM, not the BlackBox GPU fleet node: minus 300
- Speed table is one two-pass workload on one machine, with named run to run variance: minus 300
- Single artifact version (version 1) only, no multi-version migration or rollback test: minus 200
Total deductions: 2,500. Grade: 7,500 of 10,000.

## Next

Level 3, Deep: a real-time inference API and an edge deployment study per SPEC, plus the TensorFlow side when TensorFlow is available (TensorFlow Serving or TFLite deployment, one TF Profiler note, and the written judgment page on when TensorFlow is the right tool).
