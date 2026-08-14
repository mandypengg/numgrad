"""Tests for loading and batching in ``numgrad.data``.

Nothing here touches the network. ``load_mnist`` is exercised against a cache
directory the test fills itself, and the one function that would download is
replaced with a stand-in that records whether it was called. That is the point
of the cache tests: a fetch on the second call would be invisible to a test that
only looked at the arrays that came back.

The batching tests are mostly about a single failure mode. An iterator that
shuffles images and labels with two different permutations, or that drops the
short final batch, still returns arrays of the right shape and dtype, and a
model trained on it still runs. It just never learns.
"""

import numpy as np
import pytest

from numgrad import batches, load_mnist, one_hot, train_test_split
from numgrad.data import IMAGE_SIZE

# A stand-in for the 70k dataset: the same 784 columns and uint8 range, small
# enough to write to disk in a test. Each image is filled with its own index
# modulo 256, so a row can be traced back to where it started.
FAKE_ROWS = 70


def fake_dataset(rows=FAKE_ROWS):
    """Raw uint8 pixels and int64 labels, shaped like what OpenML hands back."""
    images = np.zeros((rows, IMAGE_SIZE), dtype=np.uint8)
    for row in range(rows):
        images[row] = row % 256

    # The extremes matter for the normalization test: 0 and 255 have to land on
    # exactly 0.0 and 1.0.
    images[0] = 0
    images[1] = 255

    labels = np.arange(rows, dtype=np.int64) % 10
    return images, labels


def write_cache(cache_dir, images, labels):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "mnist_images.npy", images)
    np.save(cache_dir / "mnist_labels.npy", labels)


@pytest.fixture
def small_split(monkeypatch):
    """Make ``load_mnist`` split a 70-row dataset 60/10 instead of 60k/10k.

    ``load_mnist`` reads the module-level constant at call time and passes it
    down, so patching it here is enough. The ratio is kept the same as the real
    one so the assertions below read like the real thing.
    """
    monkeypatch.setattr("numgrad.data.TRAIN_SIZE", 60)


@pytest.fixture
def no_fetch(monkeypatch):
    """Fail loudly if anything tries to download during a test."""

    def refuse():
        raise AssertionError("_fetch_mnist was called, but the cache was populated")

    monkeypatch.setattr("numgrad.data._fetch_mnist", refuse)


def test_load_mnist_reads_a_populated_cache(tmp_path, small_split, no_fetch):
    images, labels = fake_dataset()
    write_cache(tmp_path, images, labels)

    x_train, y_train, x_test, y_test = load_mnist(cache_dir=tmp_path)

    assert x_train.shape == (60, IMAGE_SIZE)
    assert y_train.shape == (60,)
    assert x_test.shape == (10, IMAGE_SIZE)
    assert y_test.shape == (10,)


def test_pixels_are_float64_in_the_unit_interval(tmp_path, small_split, no_fetch):
    images, labels = fake_dataset()
    write_cache(tmp_path, images, labels)

    x_train, y_train, _, _ = load_mnist(cache_dir=tmp_path)

    assert x_train.dtype == np.float64
    assert x_train.min() == 0.0
    assert x_train.max() == 1.0

    # The scaling is exactly /255, not a min-max rescale of whatever showed up:
    # the second row was all 255s and the first all 0s.
    np.testing.assert_array_equal(x_train[0], np.zeros(IMAGE_SIZE))
    np.testing.assert_array_equal(x_train[1], np.ones(IMAGE_SIZE))
    np.testing.assert_allclose(x_train[2], np.full(IMAGE_SIZE, 2.0 / 255.0))

    # Labels stay integers, because softmax_cross_entropy indexes with them.
    assert y_train.dtype == np.int64


def test_the_split_is_the_stored_order_not_a_shuffle(tmp_path, small_split, no_fetch):
    """Row 60 of the dataset is row 0 of the test set, every time."""
    images, labels = fake_dataset()
    write_cache(tmp_path, images, labels)

    _, y_train, _, y_test = load_mnist(cache_dir=tmp_path)

    np.testing.assert_array_equal(y_train, labels[:60])
    np.testing.assert_array_equal(y_test, labels[60:])


def counting_fetch(monkeypatch, images, labels):
    """Stand in for the download and return a list that grows once per call."""
    calls = []

    def fetch():
        calls.append(1)
        return images, labels

    monkeypatch.setattr("numgrad.data._fetch_mnist", fetch)
    return calls


def test_a_cache_miss_fetches_once_and_writes_both_files(
    tmp_path, small_split, monkeypatch
):
    images, labels = fake_dataset()
    calls = counting_fetch(monkeypatch, images, labels)

    first = load_mnist(cache_dir=tmp_path)

    assert len(calls) == 1
    assert (tmp_path / "mnist_images.npy").exists()
    assert (tmp_path / "mnist_labels.npy").exists()

    # Second call: same arrays, and no second download.
    second = load_mnist(cache_dir=tmp_path)

    assert len(calls) == 1
    for got, want in zip(second, first):
        np.testing.assert_array_equal(got, want)


def test_the_cache_stores_raw_uint8(tmp_path, small_split, monkeypatch):
    """Not the float64 the loader returns. 70k rows of float64 is 440 MB."""
    images, labels = fake_dataset()
    counting_fetch(monkeypatch, images, labels)

    load_mnist(cache_dir=tmp_path)

    assert np.load(tmp_path / "mnist_images.npy").dtype == np.uint8


def test_a_half_written_cache_is_not_read(tmp_path, small_split, monkeypatch):
    """Only the images file present, so the loader refetches rather than crashing."""
    images, labels = fake_dataset()
    np.save(tmp_path / "mnist_images.npy", images)
    calls = counting_fetch(monkeypatch, images, labels)

    x_train, _, _, _ = load_mnist(cache_dir=tmp_path)

    assert len(calls) == 1
    assert x_train.shape == (60, IMAGE_SIZE)


def test_images_of_the_wrong_width_are_rejected(tmp_path, small_split, no_fetch):
    write_cache(tmp_path, np.zeros((70, 10), dtype=np.uint8), np.zeros(70, dtype=int))

    with pytest.raises(ValueError, match=f"{IMAGE_SIZE} pixels"):
        load_mnist(cache_dir=tmp_path)


# train_test_split.


def test_train_test_split_slices_at_the_given_point():
    images = np.arange(50).reshape(10, 5).astype(np.float64)
    labels = np.arange(10)

    x_train, y_train, x_test, y_test = train_test_split(images, labels, train_size=7)

    assert x_train.shape == (7, 5)
    assert x_test.shape == (3, 5)
    np.testing.assert_array_equal(x_train, images[:7])
    np.testing.assert_array_equal(x_test, images[7:])
    np.testing.assert_array_equal(y_train, labels[:7])
    np.testing.assert_array_equal(y_test, labels[7:])


def test_train_test_split_rejects_bad_inputs():
    images = np.zeros((10, 5))

    with pytest.raises(ValueError, match="one label per image"):
        train_test_split(images, np.zeros(9), train_size=7)

    with pytest.raises(ValueError, match="train_size"):
        train_test_split(images, np.zeros(10), train_size=10)

    with pytest.raises(ValueError, match="train_size"):
        train_test_split(images, np.zeros(10), train_size=0)


# one_hot.


def test_one_hot_puts_a_single_one_in_each_row():
    encoded = one_hot(np.array([0, 3, 9]), classes=10)

    assert encoded.shape == (3, 10)
    assert encoded.dtype == np.float64
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(3))
    assert encoded[0, 0] == 1.0
    assert encoded[1, 3] == 1.0
    assert encoded[2, 9] == 1.0


def test_one_hot_rejects_out_of_range_labels():
    with pytest.raises(ValueError, match=r"\[0, 10\)"):
        one_hot(np.array([0, 10]))

    with pytest.raises(ValueError, match=r"\[0, 10\)"):
        one_hot(np.array([-1, 2]))

    with pytest.raises(ValueError, match="1-D labels"):
        one_hot(np.zeros((2, 3), dtype=int))


# batches.


def paired_data(rows=10, cols=3):
    """Images whose every entry is the row's label, so a mispairing is visible."""
    labels = np.arange(rows)
    images = np.repeat(labels.reshape(rows, 1), cols, axis=1).astype(np.float64)
    return images, labels


def label_order(iterator):
    """Every label an iterator yields, in the order it yielded them."""
    return np.concatenate([batch_labels for _, batch_labels in iterator])


def test_batches_cover_every_row_exactly_once():
    images, labels = paired_data(rows=10)

    seen = label_order(batches(images, labels, batch_size=3, seed=0))

    np.testing.assert_array_equal(np.sort(seen), labels)


def test_the_short_final_batch_is_yielded():
    """10 rows in batches of 3 is 3 + 3 + 3 + 1, not 3 + 3 + 3."""
    images, labels = paired_data(rows=10)

    sizes = [
        len(batch_labels) for _, batch_labels in batches(images, labels, 3, seed=0)
    ]

    assert sizes == [3, 3, 3, 1]


def test_shuffling_keeps_each_image_with_its_own_label():
    """The failure this whole file exists for.

    Two independent permutations would still yield the right shapes, the right
    dtypes, and every label exactly once. Every image would just be paired with
    someone else's answer, and the only symptom would be a model that never gets
    above chance.
    """
    images, labels = paired_data(rows=20, cols=4)

    for batch_images, batch_labels in batches(images, labels, batch_size=6, seed=1):
        expected = np.repeat(batch_labels.reshape(-1, 1), 4, axis=1)
        np.testing.assert_array_equal(batch_images, expected)


def test_shuffle_off_walks_the_stored_order():
    images, labels = paired_data(rows=10)

    seen = label_order(batches(images, labels, batch_size=4, shuffle=False))

    np.testing.assert_array_equal(seen, labels)


def test_the_same_seed_gives_the_same_order():
    images, labels = paired_data(rows=20)

    def order(seed):
        return label_order(batches(images, labels, batch_size=6, seed=seed))

    np.testing.assert_array_equal(order(7), order(7))
    assert not np.array_equal(order(7), order(8))

    # A pass really is shuffled, rather than a seed that quietly does nothing.
    assert not np.array_equal(order(7), labels)


def test_a_tuple_seed_reshuffles_between_epochs():
    """How the training loop gets a new order each epoch from one --seed flag."""
    images, labels = paired_data(rows=20)

    def epoch_order(seed, epoch):
        return label_order(batches(images, labels, batch_size=6, seed=(seed, epoch)))

    assert not np.array_equal(epoch_order(5, 0), epoch_order(5, 1))
    np.testing.assert_array_equal(epoch_order(5, 0), epoch_order(5, 0))


def test_batches_do_not_mutate_the_inputs():
    images, labels = paired_data(rows=10)
    images_before = images.copy()
    labels_before = labels.copy()

    for batch_images, _ in batches(images, labels, batch_size=3, seed=2):
        batch_images += 1.0

    np.testing.assert_array_equal(images, images_before)
    np.testing.assert_array_equal(labels, labels_before)


def test_a_batch_larger_than_the_data_is_one_batch():
    images, labels = paired_data(rows=5)

    sizes = [len(bl) for _, bl in batches(images, labels, batch_size=64, seed=3)]

    assert sizes == [5]


def test_batches_reject_bad_inputs():
    images, labels = paired_data(rows=10)

    with pytest.raises(ValueError, match="one label per image"):
        list(batches(images, labels[:9], batch_size=3))

    with pytest.raises(ValueError, match="batch_size"):
        list(batches(images, labels, batch_size=0))
