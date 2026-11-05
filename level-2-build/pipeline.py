"""Level 2 Build: data pipeline at scale, NumPy plus standard library only.

SPEC Build asks for a tf.data pipeline at scale (parallel maps,
prefetch, caching) with a speed table against a naive loader, plus
SavedModel discipline (handled in artifact.py).

TensorFlow was checked again for this level on 2026-11-05:
``python3 -c "import tensorflow"`` returned
ModuleNotFoundError: No module named 'tensorflow', and torch was
absent too. No large package was installed and no TensorFlow run was
faked. The tf.data ideas are therefore implemented here in labelled
NumPy and standard library equivalents:

- parallel map: ProcessPoolExecutor over signal chunks, order preserved
- caching: the first pass extracts features once; later passes reuse
  the cached feature array and extract nothing again
- prefetch style batching: a PrefetchBatches iterator prepares the
  next batch while the current batch is being consumed
- deterministic batching: the batch order comes from a seeded
  permutation, stated in code and in results_build.json

The feature extractor and the dataset generator are imported from the
Level 1 Learn code, not copied, so Build and Learn cannot diverge.
All data is synthetic, seeded, and labelled synthetic.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEARN = os.path.join(os.path.dirname(_HERE), "level-1-learn")
if _LEARN not in sys.path:
    sys.path.insert(0, _LEARN)

from signal_ml import extract_features, make_dataset  # noqa: E402

DEFAULT_SEED = 123
DEFAULT_WORKERS = 4
DEFAULT_CHUNK_SIZE = 256


def _extract_chunk(signals):
    """Top level worker function so ProcessPoolExecutor can pickle it."""
    return np.array([extract_features(s) for s in signals])


def naive_features(signals):
    """Naive loader: per-signal feature extraction in a plain loop.

    No caching, no parallelism. Calling it twice extracts twice.
    """
    return np.array([extract_features(s) for s in signals])


def make_large_dataset(n_per_class=1200, seed=0):
    """Larger synthetic dataset, seeded. Labelled synthetic only."""
    return make_dataset(n_per_class=n_per_class, seed=seed)


class OptimizedPipeline:
    """Parallel map plus cache pipeline over a fixed signal array."""

    def __init__(self, signals, labels, workers=DEFAULT_WORKERS,
                 chunk_size=DEFAULT_CHUNK_SIZE, seed=DEFAULT_SEED):
        self.signals = np.asarray(signals)
        self.labels = np.asarray(labels)
        self.workers = int(workers)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self._cached_features = None
        self.extraction_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _chunks(self):
        n = len(self.signals)
        return [self.signals[i:i + self.chunk_size]
                for i in range(0, n, self.chunk_size)]

    def load(self):
        """Return (features, labels). First call extracts, later calls cache."""
        if self._cached_features is not None:
            self.cache_hits += 1
            return self._cached_features, self.labels
        self.cache_misses += 1
        chunks = self._chunks()
        if self.workers > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                parts = list(pool.map(_extract_chunk, chunks))
        else:
            parts = [_extract_chunk(c) for c in chunks]
        features = np.vstack(parts)
        self.extraction_count += int(len(self.signals))
        self._cached_features = features
        return features, self.labels

    def clear_cache(self):
        self._cached_features = None

    def batch_indices(self, batch_size=128, shuffle=True):
        """Deterministic batch index lists from the stated seed."""
        n = len(self.labels)
        if shuffle:
            rng = np.random.default_rng(self.seed)
            order = rng.permutation(n)
        else:
            order = np.arange(n)
        return [order[i:i + batch_size].tolist()
                for i in range(0, n, batch_size)]

    def batches(self, batch_size=128, shuffle=True):
        """List of (feature_batch, label_batch) in deterministic order."""
        features, labels = self.load()
        return [(features[idx], labels[idx])
                for idx in self.batch_indices(batch_size, shuffle=shuffle)]


class PrefetchBatches:
    """Prefetch style iterator: prepares the next batch ahead of use.

    The next batch is submitted to a single worker thread while the
    current batch is yielded, mirroring tf.data prefetch in stdlib
    form. Batch content and order are identical to
    OptimizedPipeline.batches for the same seed and batch size.
    """

    def __init__(self, pipeline, batch_size=128, shuffle=True):
        self.pipeline = pipeline
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self._index_batches = pipeline.batch_indices(batch_size, shuffle=shuffle)
        self._features, self._labels = pipeline.load()

    def _make(self, idx):
        return self._features[idx], self._labels[idx]

    def __iter__(self):
        if not self._index_batches:
            return
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self._make, self._index_batches[0])
            for pos, idx in enumerate(self._index_batches):
                current = pending.result()
                if pos + 1 < len(self._index_batches):
                    pending = pool.submit(self._make, self._index_batches[pos + 1])
                yield current

    def __len__(self):
        return len(self._index_batches)
