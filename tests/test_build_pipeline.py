import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))

from artifact import artifact_dir, load_artifact, predict_with_artifact, save_artifact  # noqa: E402
from pipeline import OptimizedPipeline, PrefetchBatches, naive_features  # noqa: E402
from signal_ml import features_for_dataset, make_dataset  # noqa: E402
from train_build import train_at_scale  # noqa: E402


def small_data(n=20, seed=0):
    return make_dataset(n_per_class=n, seed=seed)


class PipelineEqualityTests(unittest.TestCase):
    def test_naive_vs_optimized_equal(self):
        X, y = small_data()
        feats_naive = naive_features(X)
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=8, seed=5)
        feats_opt, labels = pipe.load()
        self.assertTrue(np.array_equal(feats_naive, feats_opt))
        self.assertTrue(np.array_equal(y, labels))
        self.assertEqual(feats_opt.shape, (len(y), 12))

    def test_naive_matches_learn_features(self):
        X, y = small_data(n=10)
        self.assertTrue(np.array_equal(naive_features(X), features_for_dataset(X)))

    def test_parallel_order_preserved(self):
        X, y = small_data(n=30)
        pipe = OptimizedPipeline(X, y, workers=4, chunk_size=7, seed=5)
        feats, _ = pipe.load()
        self.assertTrue(np.array_equal(feats, features_for_dataset(X)))


class CacheTests(unittest.TestCase):
    def test_second_load_uses_cache(self):
        X, y = small_data()
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=8, seed=5)
        first, _ = pipe.load()
        count_after_first = pipe.extraction_count
        second, _ = pipe.load()
        self.assertEqual(pipe.extraction_count, count_after_first)
        self.assertEqual(pipe.extraction_count, len(y))
        self.assertGreaterEqual(pipe.cache_hits, 1)
        self.assertTrue(np.array_equal(first, second))

    def test_clear_cache_reextracts(self):
        X, y = small_data(n=10)
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=8, seed=5)
        pipe.load()
        pipe.clear_cache()
        pipe.load()
        self.assertEqual(pipe.extraction_count, 2 * len(y))
        self.assertEqual(pipe.cache_misses, 2)


class BatchTests(unittest.TestCase):
    def test_batch_determinism_same_seed(self):
        X, y = small_data(n=25)
        p1 = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=99)
        p2 = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=99)
        self.assertEqual(p1.batch_indices(16), p2.batch_indices(16))

    def test_batch_different_seed_differs(self):
        X, y = small_data(n=25)
        p1 = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=1)
        p2 = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=2)
        flat1 = [i for b in p1.batch_indices(16) for i in b]
        flat2 = [i for b in p2.batch_indices(16) for i in b]
        self.assertNotEqual(flat1, flat2)

    def test_batches_cover_all_rows_once(self):
        X, y = small_data(n=25)
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=3)
        idx = [i for b in pipe.batch_indices(16) for i in b]
        self.assertEqual(sorted(idx), list(range(len(y))))

    def test_batch_sizes_and_last_batch(self):
        X, y = small_data(n=25)  # 50 rows
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=3)
        batches = pipe.batches(batch_size=16)
        self.assertEqual([len(labels) for _, labels in batches], [16, 16, 16, 2])

    def test_no_shuffle_is_sequential(self):
        X, y = small_data(n=10)
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=8, seed=3)
        idx = [i for b in pipe.batch_indices(8, shuffle=False) for i in b]
        self.assertEqual(idx, list(range(len(y))))

    def test_prefetch_matches_batches(self):
        X, y = small_data(n=25)
        pipe = OptimizedPipeline(X, y, workers=2, chunk_size=16, seed=7)
        expected = pipe.batches(batch_size=16)
        got = list(PrefetchBatches(pipe, batch_size=16, shuffle=True))
        self.assertEqual(len(got), len(expected))
        for (ef, el), (gf, gl) in zip(expected, got):
            self.assertTrue(np.array_equal(ef, gf))
            self.assertTrue(np.array_equal(el, gl))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        Xtr_sig, ytr = small_data(n=30, seed=0)
        Xte_sig, yte = small_data(n=15, seed=7)
        Xtr = features_for_dataset(Xtr_sig)
        Xte = features_for_dataset(Xte_sig)
        self.Xte, self.yte = Xte, yte
        self.model, self.metrics, self.mu, self.sd = train_at_scale(
            Xtr, ytr, Xte, yte, epochs=50, lr=0.05, seed=1)
        self.config = {"epochs": 50, "lr": 0.05, "hidden": 16, "seed": 1}
        self.manifest = save_artifact(self.model, self.mu, self.sd,
                                      self.tmp, version=1, config=self.config)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip_predictions_equal(self):
        model, mu, sd, manifest = load_artifact(self.tmp, version=1)
        loaded = predict_with_artifact(model, mu, sd, self.Xte)
        direct = self.model.predict((self.Xte - self.mu) / self.sd)
        self.assertTrue(np.array_equal(loaded, direct))

    def test_manifest_fields(self):
        self.assertEqual(self.manifest["input_signature"]["feature_dim"], 12)
        self.assertEqual(self.manifest["output_signature"]["classes"], 2)
        self.assertEqual(self.manifest["version"], 1)
        self.assertIn("not SavedModel", self.manifest["format"])
        self.assertEqual(len(self.manifest["weights_sha256"]), 64)
        self.assertEqual(self.manifest["training_config"], self.config)
        with open(os.path.join(artifact_dir(self.tmp, 1), "manifest.json")) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk, self.manifest)

    def test_versioned_layout_exists(self):
        d = artifact_dir(self.tmp, 1)
        for name in ["manifest.json", "W1.npy", "b1.npy", "W2.npy", "b2.npy", "mu.npy", "sd.npy"]:
            self.assertTrue(os.path.exists(os.path.join(d, name)), name)

    def test_tampered_weights_rejected(self):
        path = os.path.join(artifact_dir(self.tmp, 1), "W1.npy")
        with open(path, "rb") as f:
            data = bytearray(f.read())
        data[-1] ^= 0xFF
        with open(path, "wb") as f:
            f.write(bytes(data))
        with self.assertRaises(ValueError):
            load_artifact(self.tmp, version=1)

    def test_tampered_manifest_hash_rejected(self):
        mpath = os.path.join(artifact_dir(self.tmp, 1), "manifest.json")
        manifest = copy.deepcopy(self.manifest)
        manifest["weights_sha256"] = "0" * 64
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        with self.assertRaises(ValueError):
            load_artifact(self.tmp, version=1)

    def test_tampered_signature_rejected(self):
        mpath = os.path.join(artifact_dir(self.tmp, 1), "manifest.json")
        manifest = copy.deepcopy(self.manifest)
        manifest["input_signature"]["feature_dim"] = 99
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        with self.assertRaises(ValueError):
            load_artifact(self.tmp, version=1)

    def test_missing_version_rejected(self):
        with self.assertRaises(ValueError):
            load_artifact(self.tmp, version=99)


class TrainingTests(unittest.TestCase):
    def test_training_accuracy_floor_and_loss_decrease(self):
        Xtr_sig, ytr = small_data(n=50, seed=0)
        Xte_sig, yte = small_data(n=25, seed=7)
        Xtr = features_for_dataset(Xtr_sig)
        Xte = features_for_dataset(Xte_sig)
        _, metrics, _, _ = train_at_scale(Xtr, ytr, Xte, yte, epochs=100, lr=0.05, seed=1)
        self.assertGreaterEqual(metrics["test_accuracy"], 0.90)
        curve = metrics["loss_curve"]
        self.assertGreater(curve[0]["loss"], curve[-1]["loss"])

    def test_training_deterministic(self):
        Xtr_sig, ytr = small_data(n=30, seed=0)
        Xte_sig, yte = small_data(n=15, seed=7)
        Xtr = features_for_dataset(Xtr_sig)
        Xte = features_for_dataset(Xte_sig)
        _, m1, _, _ = train_at_scale(Xtr, ytr, Xte, yte, epochs=30, lr=0.05, seed=1)
        _, m2, _, _ = train_at_scale(Xtr, ytr, Xte, yte, epochs=30, lr=0.05, seed=1)
        self.assertEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()
