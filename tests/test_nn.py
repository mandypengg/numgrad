"""Tests for the pieces in ``numgrad.nn``.

The losses here are composed from ops rather than carrying gradients of their
own, so the gradient checks are checking the composition: that the graph a loss
builds differentiates to the right thing, and that it stays numerically sound at
logit magnitudes where the textbook formula does not.
"""

import numpy as np
import pytest

from numgrad import Tensor, check_grads, softmax_cross_entropy


def reference_loss(logits, labels):
    """Mean cross entropy computed directly in NumPy, for comparison.

    Written the stable way, since the point of comparing against it is the value
    of the loss and not the arithmetic used to reach it.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return -np.mean(np.log(probs[np.arange(logits.shape[0]), labels]))


def reference_loss_in_log_space(logits, labels):
    """The same loss via ``np.logaddexp``, which never forms a probability.

    ``reference_loss`` divides and then logs, so a true class whose probability
    underflows to exactly 0 sends it to inf. Accumulating in log space instead
    keeps that case finite, and it is an implementation NumPy provides rather
    than a restatement of the one under test.
    """
    rows = np.arange(logits.shape[0])
    return np.mean(np.logaddexp.reduce(logits, axis=1) - logits[rows, labels])


def reference_grad(logits, labels):
    """d(loss)/d(logits), worked out by hand: (softmax - one hot) / batch."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(logits.shape[0]), labels] = 1.0

    return (probs - one_hot) / logits.shape[0]


LABELS = np.array([0, 3, 1, 2, 2, 0])


def test_forward_matches_the_reference():
    rng = np.random.default_rng(40)
    logits = Tensor(rng.standard_normal((6, 4)))

    loss = softmax_cross_entropy(logits, LABELS)

    assert loss.shape == ()
    np.testing.assert_allclose(loss.data, reference_loss(logits.data, LABELS))


def test_uniform_logits_give_log_of_the_class_count():
    """Every class equally likely, so the loss is log(classes) exactly."""
    logits = Tensor(np.zeros((3, 5)))

    loss = softmax_cross_entropy(logits, np.array([0, 2, 4]))

    np.testing.assert_allclose(loss.data, np.log(5.0))


def test_a_confident_correct_prediction_costs_almost_nothing():
    logits = Tensor([[50.0, 0.0, 0.0], [0.0, 0.0, 50.0]])

    loss = softmax_cross_entropy(logits, np.array([0, 2]))

    assert 0.0 <= float(loss.data) < 1e-15


def test_gradient_matches_the_hand_derived_form():
    """The analytic gradient is (softmax - one hot) / batch.

    Worth pinning against the closed form as well as against a finite
    difference: the two disagree in different ways, and a composition that was
    off by the batch factor would still pass a check that only compared shapes.
    """
    rng = np.random.default_rng(41)
    logits = Tensor(rng.standard_normal((6, 4)))

    softmax_cross_entropy(logits, LABELS).backward()

    np.testing.assert_allclose(logits.grad, reference_grad(logits.data, LABELS))


def test_the_rows_of_the_gradient_sum_to_zero():
    """Adding a constant to a row cannot change the loss, so each row sums to 0.

    This is the shift invariance the log-sum-exp trick relies on, read off the
    gradient. It also means the row max carries no gradient of its own, which is
    why it can be left on the tape.
    """
    rng = np.random.default_rng(42)
    logits = Tensor(rng.standard_normal((6, 4)))

    softmax_cross_entropy(logits, LABELS).backward()

    np.testing.assert_allclose(logits.grad.sum(axis=1), np.zeros(6), atol=1e-15)


def test_shifting_a_row_leaves_the_loss_unchanged():
    rng = np.random.default_rng(43)
    logits = rng.standard_normal((6, 4))
    shifts = rng.standard_normal((6, 1))

    plain = softmax_cross_entropy(Tensor(logits), LABELS)
    shifted = softmax_cross_entropy(Tensor(logits + shifts), LABELS)

    np.testing.assert_allclose(plain.data, shifted.data)


def test_gradcheck_softmax_cross_entropy():
    rng = np.random.default_rng(44)
    logits = Tensor(rng.standard_normal((6, 4)))

    check_grads(lambda x: softmax_cross_entropy(x, LABELS), [logits])


def test_gradcheck_with_every_row_sharing_a_label():
    """One class for the whole batch, so its column takes every one-hot term."""
    rng = np.random.default_rng(45)
    logits = Tensor(rng.standard_normal((6, 4)))

    check_grads(lambda x: softmax_cross_entropy(x, np.zeros(6, dtype=int)), [logits])


def test_gradcheck_on_a_single_row_batch():
    rng = np.random.default_rng(46)
    logits = Tensor(rng.standard_normal((1, 5)))

    check_grads(lambda x: softmax_cross_entropy(x, np.array([3])), [logits])


# The reason the loss subtracts the row max at all. Logits this large are not
# exotic: an untrained net with a bad initialization produces them in the first
# few steps, and the naive formula returns nan there rather than a large loss.


def test_huge_logits_stay_finite_and_differentiable():
    """Logits of magnitude 1e3, where softmax-then-log overflows.

    ``exp(1e3)`` is ``inf`` in float64, so forming the softmax first gives
    ``inf / inf``, and the nan that produces flows through the backward pass and
    destroys every gradient in the graph. Subtracting the row max caps the
    largest exponent at ``exp(0)``, so nothing here is ever exponentiated above
    1 and the loss comes out at the same size as it would for small logits.
    """
    rng = np.random.default_rng(47)
    logits = Tensor(1e3 + rng.standard_normal((6, 4)))

    # The naive path really does overflow at these inputs, which is what makes
    # this test worth having rather than a restatement of the small-logit one.
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(logits.data)).any()

    loss = softmax_cross_entropy(logits, LABELS)

    assert np.isfinite(loss.data)
    np.testing.assert_allclose(loss.data, reference_loss(logits.data, LABELS))

    # And the gradient is not merely finite, it is still correct.
    check_grads(lambda x: softmax_cross_entropy(x, LABELS), [logits])


def test_huge_logits_that_are_also_far_apart():
    """Magnitude 1e3 with a spread to match, so the probabilities underflow.

    An underflowed exp is a probability that genuinely was negligible, so the
    loss and the gradient are still right. Note which reference this compares
    against: the loss here is around 940, meaning the true class had probability
    exp(-940), which is 0 in float64. Forming that probability and then taking
    its log gives inf, so ``reference_loss`` cannot describe this case, while
    the loss under test never forms it and stays finite.

    There is no ``check_grads`` call here: the loss is of order 1e3 while the
    gradients being measured are of order 1e-8, and differencing the one to
    recover the other costs more digits than float64 has. That is a limit of
    finite differencing, not of the backward pass, so the gradient is compared
    against the closed form instead.
    """
    rng = np.random.default_rng(48)
    logits = Tensor(1e3 * rng.standard_normal((6, 4)))

    loss = softmax_cross_entropy(logits, LABELS)
    loss.backward()

    assert np.isfinite(loss.data)
    np.testing.assert_allclose(
        loss.data, reference_loss_in_log_space(logits.data, LABELS)
    )

    # The textbook formula gives up here, which is the whole point.
    with np.errstate(divide="ignore"):
        assert not np.isfinite(reference_loss(logits.data, LABELS))

    assert np.isfinite(logits.grad).all()
    # Compared with an absolute tolerance, because the entries that disagree
    # with the reference at all disagree at magnitudes around 1e-191: a class
    # whose probability underflowed contributes a gradient far below anything
    # that could matter, and the two routes to it round differently. A relative
    # comparison would call 0 against 1e-191 an infinite error.
    np.testing.assert_allclose(
        logits.grad, reference_grad(logits.data, LABELS), atol=1e-12
    )


def test_very_negative_logits_stay_finite():
    """The other end of the range, where exp underflows to 0 rather than to inf.

    The true class still has to keep its log-probability finite: the shifted max
    is exactly 0, so the row sum is at least 1 and its log cannot diverge.
    """
    logits = Tensor(-1e3 + np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]))

    loss = softmax_cross_entropy(logits, np.array([0, 2]))

    assert np.isfinite(loss.data)
    np.testing.assert_allclose(loss.data, reference_loss(logits.data, np.array([0, 2])))


def test_loss_and_gradients_are_float64():
    logits = Tensor([[1, 2, 3], [4, 5, 6]])

    loss = softmax_cross_entropy(logits, np.array([0, 2]))
    loss.backward()

    assert loss.data.dtype == np.float64
    assert logits.grad.dtype == np.float64


def test_the_input_logits_are_not_mutated():
    """The row max is subtracted into a new tensor, not in place."""
    rng = np.random.default_rng(49)
    logits = Tensor(rng.standard_normal((6, 4)))
    before = logits.data.copy()

    softmax_cross_entropy(logits, LABELS).backward()

    np.testing.assert_array_equal(logits.data, before)


def test_wrong_shapes_are_rejected():
    logits = Tensor(np.zeros((6, 4)))

    with pytest.raises(ValueError, match="2-D logits"):
        softmax_cross_entropy(Tensor(np.zeros(4)), np.array([0]))

    with pytest.raises(ValueError, match="one label per row"):
        softmax_cross_entropy(logits, np.array([0, 1]))
