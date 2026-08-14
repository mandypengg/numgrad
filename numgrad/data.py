"""Dataset loading and batching.

MNIST is fetched with ``sklearn.datasets.fetch_openml``. That is the only use of
scikit-learn in this project — it is a download helper, nothing more. No
sklearn model, metric, or preprocessing utility belongs here. The import sits
inside the fetch function rather than at module scope, so importing ``numgrad``
never requires sklearn to be installed and a cache hit never touches it.

Pixels arrive as uint8 and are cast to ``float64`` and scaled to [0, 1] on the
way in. Labels stay integer class indices, because that is what
``softmax_cross_entropy`` indexes with; ``one_hot`` is here for anything that
wants the ``float64`` matrix instead.

Implemented so far: load_mnist, train_test_split, one_hot, batches.
"""

from pathlib import Path

import numpy as np

# 28x28 pixels flattened into a row, ten digit classes.
IMAGE_SIZE = 784
CLASSES = 10

# OpenML ships MNIST as one 70k block whose last 10k rows are the canonical test
# set. The split is a fixed slice, not a random partition, so that every report
# of an accuracy number is comparable with every other one.
TRAIN_SIZE = 60_000

# Sits next to the package rather than in the user's home directory, so a reader
# can see what was downloaded. It is in .gitignore.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch_mnist():
    """Download MNIST from OpenML, returning raw uint8 pixels and int labels."""
    # Imported here, so this is the only line in the project that needs sklearn
    # present, and it only runs on a cache miss.
    from sklearn.datasets import fetch_openml

    dataset = fetch_openml("mnist_784", version=1, as_frame=False)

    # Pixels are integers in [0, 255] regardless of the dtype sklearn hands
    # back, and the targets are digit strings like '5'.
    images = np.asarray(dataset.data).astype(np.uint8)
    labels = np.asarray(dataset.target).astype(np.int64)

    return images, labels


def load_mnist(cache_dir=None):
    """Load MNIST, downloading it once and reading from disk thereafter.

    Parameters
    ----------
    cache_dir:
        Directory holding the cached ``.npy`` arrays. Defaults to ``data/``
        beside the package.

    Returns
    -------
    tuple
        ``(x_train, y_train, x_test, y_test)``. Images are ``float64`` in
        [0, 1] with shape ``(n, 784)``; labels are ``int64`` class indices with
        shape ``(n,)``.

    The cache stores the pixels as uint8, which is how they arrive, rather than
    as the float64 the rest of the library works in. That is the difference
    between a 55 MB file and a 440 MB one, and the cast on the way out costs a
    fraction of a second.
    """
    cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    images_path = cache_dir / "mnist_images.npy"
    labels_path = cache_dir / "mnist_labels.npy"

    # Both files or neither: a half-written cache from an interrupted download
    # would otherwise be read back as a complete one.
    if images_path.exists() and labels_path.exists():
        images = np.load(images_path)
        labels = np.load(labels_path)
    else:
        images, labels = _fetch_mnist()
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(images_path, images)
        np.save(labels_path, labels)

    if images.shape[1] != IMAGE_SIZE:
        raise ValueError(
            f"expected {IMAGE_SIZE} pixels per image, got {images.shape[1]}"
        )

    # The cast is explicit at this boundary rather than left to NumPy's dtype
    # inference downstream: everything past this line is float64.
    images = images.astype(np.float64) / 255.0
    labels = labels.astype(np.int64)

    # Passed explicitly rather than left to the default, so the split point is
    # visible at the call site.
    return train_test_split(images, labels, TRAIN_SIZE)


def train_test_split(images, labels, train_size=TRAIN_SIZE):
    """Split into the standard first-60k / last-10k halves.

    Deliberately not a random split. The rows are already in the order OpenML
    ships them, and that order is what makes 60k/10k the same benchmark everyone
    else reports.
    """
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"expected one label per image: {images.shape[0]} images, "
            f"{labels.shape[0]} labels"
        )
    if not 0 < train_size < images.shape[0]:
        raise ValueError(
            f"train_size must be between 1 and {images.shape[0] - 1}, got {train_size}"
        )

    x_train = images[:train_size]
    y_train = labels[:train_size]
    x_test = images[train_size:]
    y_test = labels[train_size:]

    return x_train, y_train, x_test, y_test


def one_hot(labels, classes=CLASSES):
    """Turn ``(n,)`` integer labels into an ``(n, classes)`` float64 matrix."""
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(f"expected 1-D labels, got shape {labels.shape}")
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError(
            f"labels must be in [0, {classes}), got range "
            f"[{labels.min()}, {labels.max()}]"
        )

    encoded = np.zeros((labels.shape[0], classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def batches(images, labels, batch_size, shuffle=True, seed=None):
    """Yield ``(image_batch, label_batch)`` pairs covering the data once.

    Parameters
    ----------
    images, labels:
        Arrays with the same length along the first axis.
    batch_size:
        Rows per batch. The last batch is short when the length is not a
        multiple of it, and it is yielded rather than dropped: those rows are as
        real as any other, and a loss averaged over the batch handles a short
        one correctly on its own.
    shuffle:
        Whether to visit the rows in a random order. On by default for training;
        pass False to walk the data in its stored order, which is what makes an
        evaluation pass reproducible without a seed.
    seed:
        Passed straight to ``np.random.default_rng``, so an int gives a fixed
        permutation. One call to this function is one pass over the data, which
        means the same seed gives the same order every time it is called. To
        reshuffle between epochs, vary the seed per epoch: ``seed=(seed, epoch)``
        is a valid seed to ``default_rng`` and gives an independent stream.

    The permutation is materialized up front and sliced, rather than the arrays
    themselves being shuffled. Nothing here mutates its inputs, and each yielded
    batch is a fresh contiguous copy that the caller can hold onto.
    """
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"expected one label per image: {images.shape[0]} images, "
            f"{labels.shape[0]} labels"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    order = np.arange(images.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(order)

    for start in range(0, order.shape[0], batch_size):
        index = order[start : start + batch_size]
        yield images[index], labels[index]
