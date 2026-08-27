"""Finite-difference gradient checks, and checks on the checker itself.

Every op in ``numgrad.ops`` gets a ``check_grads`` call here, added in the same
commit as the op itself.

The second half of the file tests ``check_grads`` in the other direction. A
checker that never fails would let every one of the tests above pass while the
backward passes were wrong, so each of the common failure modes gets an op that
is deliberately broken in that way and must be caught: a sign error, a missing
transpose, and a gradient that is off by a small factor rather than a large one.
"""

import numpy as np
import pytest

from numgrad import Tensor, check_grads, exp, log, relu, tanh


def away_from_zero(values, margin=0.5):
    """Push samples out of a band around 0, keeping their signs.

    For ops whose derivative is undefined or unstable at 0. Shifting rather than
    resampling keeps the inputs deterministic for a given seed.
    """
    signs = np.where(values >= 0.0, 1.0, -1.0)
    return signs * (np.abs(values) + margin)


def test_gradcheck_add():
    rng = np.random.default_rng(0)
    a = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x, y: x + y, [a, b])


def test_gradcheck_add_broadcast():
    rng = np.random.default_rng(1)
    a = Tensor(rng.standard_normal((4, 3)))
    # Row vector, stretched down the rows of a.
    b = Tensor(rng.standard_normal((1, 3)))

    check_grads(lambda x, y: x + y, [a, b])


def test_gradcheck_mul():
    rng = np.random.default_rng(2)
    a = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x, y: x * y, [a, b])


def test_gradcheck_mul_broadcast():
    rng = np.random.default_rng(3)
    a = Tensor(rng.standard_normal((4, 3)))
    # Column vector, stretched across the columns of a.
    b = Tensor(rng.standard_normal((4, 1)))

    check_grads(lambda x, y: x * y, [a, b])


def test_gradcheck_matmul():
    rng = np.random.default_rng(4)
    a = Tensor(rng.standard_normal((3, 5)))
    b = Tensor(rng.standard_normal((5, 2)))

    check_grads(lambda x, y: x @ y, [a, b])


def test_gradcheck_transpose():
    rng = np.random.default_rng(5)
    a = Tensor(rng.standard_normal((3, 5)))
    b = Tensor(rng.standard_normal((3, 5)))

    # Multiplied by a second input so the transpose is not the whole graph.
    check_grads(lambda x, y: (x * y).T, [a, b])


def test_gradcheck_sub():
    rng = np.random.default_rng(12)
    a = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x, y: x - y, [a, b])


def test_gradcheck_sub_broadcast():
    rng = np.random.default_rng(13)
    a = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(rng.standard_normal((3,)))

    # Subtraction is the op where a sign error hides best, so check it on the
    # subtrahend side too, where the broadcast and the minus sign compose.
    check_grads(lambda x, y: x - y, [a, b])
    check_grads(lambda x, y: y - x, [a, b])


def test_gradcheck_neg():
    rng = np.random.default_rng(14)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: -x, [a])


def test_gradcheck_truediv():
    rng = np.random.default_rng(15)
    a = Tensor(rng.standard_normal((4, 3)))
    # Denominator held away from zero: 1/b and a/b**2 both blow up near it, and
    # a finite difference across a near-singularity is meaningless.
    b = Tensor(away_from_zero(rng.standard_normal((4, 3))))

    check_grads(lambda x, y: x / y, [a, b])


def test_gradcheck_truediv_broadcast():
    rng = np.random.default_rng(16)
    a = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(away_from_zero(rng.standard_normal((4, 1))))

    check_grads(lambda x, y: x / y, [a, b])


def test_gradcheck_pow_integer_exponent():
    rng = np.random.default_rng(17)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: x**3, [a])


def test_gradcheck_pow_fractional_exponent():
    rng = np.random.default_rng(18)
    # Positive base: a fractional power of a negative number is not real.
    a = Tensor(np.abs(rng.standard_normal((4, 3))) + 0.5)

    check_grads(lambda x: x**0.5, [a])


def test_gradcheck_exp():
    rng = np.random.default_rng(19)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(exp, [a])


def test_gradcheck_log():
    rng = np.random.default_rng(20)
    # Positive and away from zero, where log and its derivative both diverge.
    a = Tensor(np.abs(rng.standard_normal((4, 3))) + 0.5)

    check_grads(log, [a])


def test_gradcheck_relu():
    """relu is not differentiable at 0, so no input is seeded near it.

    The backward pass takes the subgradient 0 at exactly 0, while a central
    difference straddling 0 measures 1/2, and one straddling a point within h of
    0 measures something in between. Neither disagreement is a bug in the
    backward pass, so the inputs are pushed out of a band around 0 rather than
    the tolerance being loosened to hide it.
    """
    rng = np.random.default_rng(21)
    a = Tensor(away_from_zero(rng.standard_normal((4, 3))))

    # No entry is anywhere near the kink, so the check is measuring the two
    # linear pieces and nothing else.
    assert np.all(np.abs(a.data) > 0.1)

    check_grads(relu, [a])


def test_gradcheck_tanh():
    rng = np.random.default_rng(22)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(tanh, [a])


def test_gradcheck_sum_all():
    rng = np.random.default_rng(23)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: x.sum(), [a])


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_gradcheck_sum_axis(axis, keepdims):
    rng = np.random.default_rng(24)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: x.sum(axis=axis, keepdims=keepdims), [a])


def test_gradcheck_max_all():
    rng = np.random.default_rng(25)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: x.max(), [a])


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_gradcheck_max_axis(axis, keepdims):
    rng = np.random.default_rng(26)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: x.max(axis=axis, keepdims=keepdims), [a])


@pytest.mark.parametrize("keepdims", [False, True])
def test_gradcheck_reduce_over_several_axes(keepdims):
    """A tuple axis reduces more than one axis at once.

    Worth its own check because the backward pass reinserts the reduced axes
    with a single expand_dims call, and it has to put back all of them.
    """
    rng = np.random.default_rng(35)
    a = Tensor(rng.standard_normal((3, 4, 2)))

    check_grads(lambda x: x.sum(axis=(0, 2), keepdims=keepdims), [a])
    check_grads(lambda x: x.max(axis=(0, 2), keepdims=keepdims), [a])


def test_gradcheck_reshape():
    rng = np.random.default_rng(27)
    a = Tensor(rng.standard_normal((4, 3)))
    b = Tensor(rng.standard_normal((4, 3)))

    # Multiplied first so the reshape is not the whole graph.
    check_grads(lambda x, y: (x * y).reshape((2, 6)), [a, b])


def test_gradcheck_getitem():
    rng = np.random.default_rng(28)
    a = Tensor(rng.standard_normal((4, 3)))

    check_grads(lambda x: x[1:3], [a])


def test_gradcheck_getitem_repeats_an_index():
    """Row 0 is selected twice, so its gradient must be the sum of both uses."""
    rng = np.random.default_rng(29)
    a = Tensor(rng.standard_normal((4, 3)))
    rows = np.array([0, 0, 2])

    check_grads(lambda x: x[rows], [a])


def test_gradcheck_getitem_by_label():
    """The indexing pattern cross-entropy needs: one logit per row."""
    rng = np.random.default_rng(30)
    logits = Tensor(rng.standard_normal((8, 4)))
    labels = np.array([0, 3, 1, 1, 2, 0, 3, 2])
    rows = np.arange(logits.shape[0])

    check_grads(lambda x: x[rows, labels], [logits])


# Broadcasting checks at the shapes a batch of logits actually produces. These
# are the three cases unbroadcast has to get right: a bias row added to a batch,
# a per-row scale, and two operands that both stretch.


def test_gradcheck_broadcast_batch_plus_bias():
    """(32, 10) + (10,): b gains a leading axis, which is summed away."""
    rng = np.random.default_rng(31)
    a = Tensor(rng.standard_normal((32, 10)))
    b = Tensor(rng.standard_normal((10,)))

    check_grads(lambda x, y: x + y, [a, b])


def test_gradcheck_broadcast_batch_times_column():
    """(32, 10) * (32, 1): b's second axis is stretched, then summed back to 1."""
    rng = np.random.default_rng(32)
    a = Tensor(rng.standard_normal((32, 10)))
    b = Tensor(rng.standard_normal((32, 1)))

    check_grads(lambda x, y: x * y, [a, b])


def test_gradcheck_broadcast_row_plus_column():
    """(1, 10) + (32, 1): both operands stretch, on different axes.

    The output is (32, 10) and neither input has that shape, so a backward pass
    that reduced only one side, or reduced the wrong axis, still produces a
    conformable array and has to be caught numerically.
    """
    rng = np.random.default_rng(33)
    a = Tensor(rng.standard_normal((1, 10)))
    b = Tensor(rng.standard_normal((32, 1)))

    check_grads(lambda x, y: x + y, [a, b])


def test_gradcheck_broadcast_through_a_nonlinearity():
    """The same stretch, with tanh on top so the gradient is not constant.

    Gradients of a bare sum are all ones, which hides an unbroadcast that sums
    the right number of entries from the wrong axis. tanh rather than relu here
    because relu's kink at 0 would need the inputs seeded away from it, and the
    point of this test is the broadcast, not the nonlinearity.
    """
    rng = np.random.default_rng(34)
    a = Tensor(rng.standard_normal((1, 10)))
    b = Tensor(rng.standard_normal((32, 1)))

    check_grads(lambda x, y: tanh(x + y), [a, b])
    check_grads(lambda x, y: tanh(x * y), [a, b])


def test_gradcheck_diamond():
    """x feeds two branches that recombine, so its gradient must sum both."""
    rng = np.random.default_rng(6)
    x = Tensor(rng.standard_normal((4, 3)))
    w = Tensor(rng.standard_normal((4, 3)))

    def build(x, w):
        left = x * w
        right = x + w
        return left + right

    check_grads(build, [x, w])


def test_gradcheck_repeated_use_deeper_in_the_graph():
    """x reaching the output by two separate routes sums both."""
    rng = np.random.default_rng(8)
    x = Tensor(rng.standard_normal((3, 3)))

    def build(x):
        squared = x * x
        return squared + x

    check_grads(build, [x])


def test_diamond_accumulates_from_both_paths():
    """The same check as above, against a gradient worked out by hand.

    With ``out = x * w + (x + w)``, d(sum(out))/dx is ``w + 1`` elementwise.
    A backward pass that overwrote instead of accumulating would produce either
    ``w`` or ``1`` here, depending on which branch ran last.
    """
    rng = np.random.default_rng(7)
    x = Tensor(rng.standard_normal((4, 3)))
    w = Tensor(rng.standard_normal((4, 3)))

    out = (x * w) + (x + w)
    out.backward(np.ones_like(out.data))

    np.testing.assert_allclose(x.grad, w.data + 1.0)
    np.testing.assert_allclose(w.grad, x.data + 1.0)


def test_tensor_used_twice_in_one_expression():
    """x + x must give a gradient of 2, not 1."""
    x = Tensor([3.0, -1.0])

    out = x + x
    out.backward(np.ones_like(out.data))

    np.testing.assert_allclose(x.grad, [2.0, 2.0])


def test_gradcheck_vector_jacobian_product():
    """A seed of v gives v @ J, the gradient of the scalar ``sum(v * f(x))``.

    Differenced against that scalar directly rather than through
    ``check_grads``, which always seeds with ones and so only ever measures the
    gradient of a plain sum.
    """
    rng = np.random.default_rng(11)
    x = Tensor(rng.standard_normal((3, 4)))
    v = rng.standard_normal((3, 4))

    out = tanh(x)
    out.backward(v)

    h = 1e-5
    numeric = np.zeros_like(x.data)
    for index in np.ndindex(x.data.shape):
        original = x.data[index]

        x.data[index] = original + h
        plus = float(np.sum(v * np.tanh(x.data)))

        x.data[index] = original - h
        minus = float(np.sum(v * np.tanh(x.data)))

        x.data[index] = original
        numeric[index] = (plus - minus) / (2.0 * h)

    np.testing.assert_allclose(x.grad, numeric, rtol=1e-6)


def test_gradcheck_a_seed_of_ones_is_the_gradient_of_the_sum():
    """The default question, asked explicitly, is the one check_grads asks."""
    rng = np.random.default_rng(12)
    x = Tensor(rng.standard_normal((3, 4)))

    out = exp(x)
    out.backward(np.ones_like(out.data))

    # d(sum(exp(x)))/dx is exp(x) elementwise.
    np.testing.assert_allclose(x.grad, np.exp(x.data))


def test_a_one_hot_seed_extracts_one_row_of_the_jacobian():
    """Seeding with a single 1 asks for the gradient of that one output entry.

    Run once per output entry, this is how reverse mode builds a full Jacobian,
    and it is why the seed is a parameter rather than always ones.
    """
    rng = np.random.default_rng(13)
    x = Tensor(rng.standard_normal((2, 3)))
    w = Tensor(rng.standard_normal((3, 4)))

    jacobian_rows = []
    for index in np.ndindex((2, 4)):
        x.zero_grad()
        w.zero_grad()

        seed = np.zeros((2, 4))
        seed[index] = 1.0

        (x @ w).backward(seed)
        jacobian_rows.append(x.grad.copy())

    # For out = x @ w, d out[i, j] / d x[i, :] is w[:, j], and every other row
    # of x is untouched.
    for position, index in enumerate(np.ndindex((2, 4))):
        row, column = index
        expected = np.zeros((2, 3))
        expected[row] = w.data[:, column]
        np.testing.assert_allclose(jacobian_rows[position], expected)


# Deliberately broken ops. Each one computes the correct forward value, so the
# only thing that can catch it is the gradient check.


def wrong_sign_add(a, b):
    """add, with the sign of a's gradient flipped."""
    out = Tensor(a.data + b.data, _prev=(a, b), _op="wrong_sign_add")

    def _backward():
        # Should be += out.grad. The classic sign slip.
        a.grad += -out.grad
        b.grad += out.grad

    out._backward = _backward
    return out


def wrong_matmul(a, b):
    """matmul, with the transpose missing from a's gradient.

    Both operands are square in the test below, so the missing transpose still
    produces a conformable product. That is exactly what makes this bug worth a
    test: it does not raise, it just returns the wrong numbers.
    """
    out = Tensor(a.data @ b.data, _prev=(a, b), _op="wrong_matmul")

    def _backward():
        # Should be out.grad @ b.data.T.
        a.grad += out.grad @ b.data
        b.grad += a.data.T @ out.grad

    out._backward = _backward
    return out


def slightly_wrong_mul(a, b):
    """mul, with a's gradient scaled by 1.01.

    A one percent error is far too small to notice by eye in a printed array,
    and is the size of mistake a dropped constant factor tends to produce.
    """
    out = Tensor(a.data * b.data, _prev=(a, b), _op="slightly_wrong_mul")

    def _backward():
        a.grad += 1.01 * out.grad * b.data
        b.grad += out.grad * a.data

    out._backward = _backward
    return out


def test_catches_a_sign_error():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    with pytest.raises(AssertionError) as excinfo:
        check_grads(wrong_sign_add, [a, b])

    message = str(excinfo.value)
    # The report has to say which op, which entry, and what the two values were.
    assert "wrong_sign_add" in message
    assert "index (0, 0)" in message
    assert "numeric  = 1.0" in message
    assert "analytic = -1.0" in message


def test_catches_a_missing_transpose():
    rng = np.random.default_rng(9)
    # Square, so the wrong backward pass is still a legal matrix product.
    a = Tensor(rng.standard_normal((4, 4)))
    b = Tensor(rng.standard_normal((4, 4)))

    with pytest.raises(AssertionError, match="wrong_matmul"):
        check_grads(wrong_matmul, [a, b])


def test_catches_a_one_percent_error():
    rng = np.random.default_rng(10)
    a = Tensor(rng.standard_normal((3, 3)))
    b = Tensor(rng.standard_normal((3, 3)))

    with pytest.raises(AssertionError, match="slightly_wrong_mul"):
        check_grads(slightly_wrong_mul, [a, b])


def test_correct_op_passes_where_the_broken_one_fails():
    """The two ops differ only in the backward pass, so the check is the cause.

    Without this, a failing check on wrong_sign_add could be blamed on the
    inputs or on the checker being broken outright rather than on the gradient.
    """
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    check_grads(lambda x, y: x + y, [a, b])

    with pytest.raises(AssertionError):
        check_grads(wrong_sign_add, [a, b])


def test_tolerance_is_honoured():
    """A loose enough tolerance lets the broken gradient through.

    The sign flip gives a relative error of exactly 1.0, since numeric and
    analytic are equal in magnitude and opposite in sign. Passing at tol=1.0 and
    failing at the default is proof that the comparison is actually reading tol
    rather than deciding some other way.
    """
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    check_grads(wrong_sign_add, [a, b], tol=1.0)

    with pytest.raises(AssertionError):
        check_grads(wrong_sign_add, [a, b])


def test_zero_gradients_do_not_trip_the_floor():
    """b's gradient is zero everywhere, where a naive relative error blows up.

    Dividing by |num| + |ana| alone would be 0/0 here. The 1e-8 floor in the
    denominator is what keeps this case from reporting a spurious failure.
    """
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    # b is multiplied by zero, so every entry of its gradient is exactly zero.
    check_grads(lambda x, y: x + y * 0.0, [a, b])


def test_checks_every_entry_not_just_the_first():
    """A gradient wrong in one entry only is still caught.

    An implementation that spot-checked a single index, or that stopped at the
    first agreement, would pass this.
    """

    def wrong_in_one_place(a):
        out = Tensor(a.data * 2.0, _prev=(a,), _op="wrong_in_one_place")

        def _backward():
            contribution = 2.0 * out.grad
            # Correct everywhere except the last entry.
            contribution[-1] = 0.0
            a.grad += contribution

        out._backward = _backward
        return out

    a = Tensor([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(AssertionError, match=r"index \(3,\)"):
        check_grads(wrong_in_one_place, [a])


def test_params_are_left_unmodified():
    """Every perturbation is restored, so a check does not disturb its inputs."""
    rng = np.random.default_rng(11)
    a = Tensor(rng.standard_normal((3, 2)))
    b = Tensor(rng.standard_normal((3, 2)))

    before_a = a.data.copy()
    before_b = b.data.copy()

    check_grads(lambda x, y: x * y, [a, b])

    np.testing.assert_array_equal(a.data, before_a)
    np.testing.assert_array_equal(b.data, before_b)


def test_stale_gradients_are_cleared_first():
    """A leftover gradient from an earlier backward pass must not be counted.

    Gradients accumulate, so a check that ran backward() on top of an existing
    .grad would compare the sum of two passes against one finite difference.
    """
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    # Leaves a.grad and b.grad at ones before the check begins.
    seeded = a + b
    seeded.backward(np.ones_like(seeded.data))
    np.testing.assert_array_equal(a.grad, np.ones((2, 2)))

    check_grads(lambda x, y: x + y, [a, b])
