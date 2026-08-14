"""Tape mechanics: construction, the graph, and the shape of backward().

Gradient values are checked against finite differences in ``test_gradcheck.py``.
What is tested here is the bookkeeping around them.
"""

import sys

import numpy as np
import pytest

from numgrad import Tensor, ops


def test_data_is_float64():
    x = Tensor([1, 2, 3])

    assert x.data.dtype == np.float64


def test_grad_starts_as_zeros_of_the_same_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert x.grad.shape == x.data.shape
    assert x.grad.dtype == np.float64
    np.testing.assert_array_equal(x.grad, np.zeros((2, 2)))


def test_shape():
    assert Tensor(3.0).shape == ()
    assert Tensor([1.0, 2.0]).shape == (2,)
    assert Tensor([[1.0, 2.0, 3.0]]).shape == (1, 3)


def test_transpose_property():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert x.T.shape == (3, 2)
    np.testing.assert_array_equal(x.T.data, x.data.T)


def test_transpose_is_tracked():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    out = x.T
    out.backward()

    np.testing.assert_array_equal(x.grad, np.ones((2, 2)))


def test_relu_method_is_the_op():
    """The method is a spelling of ops.relu, not a second implementation."""
    x = Tensor([[-2.0, 0.0, 3.0]])

    out = x.relu()

    assert out._op == "relu"
    np.testing.assert_array_equal(out.data, ops.relu(x).data)


def test_relu_method_is_tracked():
    x = Tensor([-2.0, 3.0])

    x.relu().backward()

    # Gradient passes where the input was positive and is stopped where it was
    # not, which is the op's rule and not the method's.
    np.testing.assert_array_equal(x.grad, [0.0, 1.0])


def test_repr_shows_shape_and_op():
    x = Tensor(np.zeros((2, 3)))
    y = Tensor(np.zeros((2, 3)))

    assert repr(x) == "Tensor(shape=(2, 3))"
    assert repr(x + y) == "Tensor(shape=(2, 3), op='add')"
    assert repr(x * y) == "Tensor(shape=(2, 3), op='mul')"


def test_leaf_has_no_parents_and_no_op():
    x = Tensor([1.0])

    assert x._prev == set()
    assert x._op == ""


def test_op_records_its_inputs():
    x = Tensor([1.0])
    y = Tensor([2.0])

    out = x + y

    assert out._prev == {x, y}
    assert out._op == "add"


def test_zero_grad_clears_in_place():
    x = Tensor([1.0, 2.0])
    y = x + x
    y.backward()

    before = x.grad
    x.zero_grad()

    np.testing.assert_array_equal(x.grad, [0.0, 0.0])
    # Same array object, so anything already holding a reference sees the reset.
    assert x.grad is before


def test_backward_seeds_the_output_with_ones():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0, 6.0], [7.0, 8.0]])

    out = x + y
    out.backward()

    np.testing.assert_array_equal(out.grad, np.ones((2, 2)))


def test_scalar_operands_are_wrapped():
    x = Tensor([1.0, 2.0])

    out = 3.0 * x + 1.0
    out.backward()

    np.testing.assert_array_equal(out.data, [4.0, 7.0])
    np.testing.assert_array_equal(x.grad, [3.0, 3.0])


def test_matmul_rejects_non_2d():
    x = Tensor([1.0, 2.0])
    y = Tensor([[1.0], [2.0]])

    with pytest.raises(ValueError):
        x @ y


def test_deep_chain_does_not_hit_the_recursion_limit():
    """A 5000-node chain is far past Python's default recursion limit.

    A recursive topological sort would raise RecursionError here rather than
    return a wrong answer, so this test fails loudly if backward() stops being
    iterative.
    """
    # Well past the default limit of 1000. Read rather than hard-coded, since a
    # test run that raised the limit would weaken the check.
    depth = max(5000, sys.getrecursionlimit() * 5)

    x = Tensor([2.0])
    y = x
    for _ in range(depth):
        y = y + x

    y.backward()

    # y is (depth + 1) * x, so the gradient is one contribution per use of x.
    np.testing.assert_allclose(y.data, [2.0 * (depth + 1)])
    np.testing.assert_allclose(x.grad, [float(depth + 1)])


def test_deep_chain_of_mixed_ops():
    """Same depth, alternating ops, so the order of the traversal matters.

    Each step computes ``y = y * a + b``. Evaluating a node before one of its
    consumers has finished accumulating would leave the chain short of its
    full gradient.
    """
    depth = 5000

    a = Tensor([1.0])
    b = Tensor([1.0])

    y = Tensor([0.0])
    for _ in range(depth):
        y = y * a + b

    y.backward()

    # With a = 1 and b = 1 the recurrence is y_n = y_{n-1} + 1, so y = depth.
    np.testing.assert_allclose(y.data, [float(depth)])
    # d y_n / db = 1 at every step, and there are `depth` steps.
    np.testing.assert_allclose(b.grad, [float(depth)])
    # d y_n / da = y_{n-1}, so the total is 0 + 1 + ... + (depth - 1).
    np.testing.assert_allclose(a.grad, [float(depth * (depth - 1) // 2)])
