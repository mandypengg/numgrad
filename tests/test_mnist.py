"""End to end on the real dataset, with the real training loop.

This is the only slow test in the suite, and the only one that will download
anything: ``load_mnist`` fetches 15 MB from OpenML on a cold cache. CI runs
``pytest -m "not slow"`` and skips it. Everything it covers is covered in
isolation elsewhere, on synthetic data, with no network.

What it adds is the join. The gradient checks prove each backward pass is right,
and the loader tests prove the batches are paired correctly, but neither notices
if the pieces are wired together wrong. A net that trains to 90 percent on real
digits is evidence that they are not, of a kind that no unit test gives.

The accuracy floor below is deliberately far under what the run actually
reaches. It is there to catch a break, not to pin a number: a regression that
matters here turns 93 percent into 11 percent, and anything that only moves the
third digit is noise this test should ignore.
"""

import numpy as np
import pytest

from numgrad import (
    SGD,
    Linear,
    ReLU,
    Sequential,
    Tensor,
    batches,
    load_mnist,
    softmax_cross_entropy,
)

# Applies to every test in this module.
pytestmark = pytest.mark.slow

# A slice of the training set rather than all of it. The point is that learning
# happens at all, and 6000 examples show that in about a second.
TRAIN_ROWS = 6000
TEST_ROWS = 2000
BATCH_SIZE = 64

# The run reaches about 88 percent at these constants. The floor is well under
# that on purpose: a regression that matters here turns 88 into 11, and a couple
# of points either way is noise this test should not have an opinion about.
ACCURACY_FLOOR = 0.80


@pytest.fixture(scope="module")
def mnist():
    """The real dataset, loaded once for the whole module."""
    return load_mnist()


def test_the_dataset_arrives_as_documented(mnist):
    x_train, y_train, x_test, y_test = mnist

    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert x_test.shape == (10000, 784)
    assert y_test.shape == (10000,)

    assert x_train.dtype == np.float64
    assert y_train.dtype == np.int64

    # Normalized, not standardized: pixels land in [0, 1] and the extremes are
    # both hit, since MNIST digits have saturated black and white.
    assert x_train.min() == 0.0
    assert x_train.max() == 1.0

    # All ten digits on both sides of the split.
    np.testing.assert_array_equal(np.unique(y_train), np.arange(10))
    np.testing.assert_array_equal(np.unique(y_test), np.arange(10))


def test_a_short_run_learns_to_read_digits(mnist):
    """The whole library in one pass: tape, ops, layers, loss, optimizer, loader.

    Deterministic from the two seeds, so this either passes at these constants
    or it fails. There is no tolerance here that a lucky shuffle could carry.
    """
    x_train, y_train, x_test, y_test = mnist
    x_train, y_train = x_train[:TRAIN_ROWS], y_train[:TRAIN_ROWS]
    x_test, y_test = x_test[:TEST_ROWS], y_test[:TEST_ROWS]

    rng = np.random.default_rng(0)
    model = Sequential(Linear(784, 128, rng=rng), ReLU(), Linear(128, 10, rng=rng))
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

    def evaluate():
        logits = model(Tensor(x_test))
        return float(np.mean(logits.data.argmax(axis=1) == y_test))

    before = evaluate()

    for batch_images, batch_labels in batches(
        x_train, y_train, BATCH_SIZE, seed=(0, 0)
    ):
        optimizer.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(batch_images)), batch_labels)
        loss.backward()
        optimizer.step()

    after = evaluate()

    # An untrained net is at chance, give or take which class its random weights
    # happen to favor. Asserted so that the floor below cannot be met by a model
    # that was already there before training.
    assert before < 0.25

    assert after > ACCURACY_FLOOR
    assert after > before + 0.5
