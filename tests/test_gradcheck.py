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

from numgrad import Tensor, check_grads


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
    out.backward()

    np.testing.assert_allclose(x.grad, w.data + 1.0)
    np.testing.assert_allclose(w.grad, x.data + 1.0)


def test_tensor_used_twice_in_one_expression():
    """x + x must give a gradient of 2, not 1."""
    x = Tensor([3.0, -1.0])

    out = x + x
    out.backward()

    np.testing.assert_allclose(x.grad, [2.0, 2.0])


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
    (a + b).backward()
    np.testing.assert_array_equal(a.grad, np.ones((2, 2)))

    check_grads(lambda x, y: x + y, [a, b])
