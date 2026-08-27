"""Forward values, shapes, and the parts of the ops a gradient check cannot see.

Gradient values live in ``test_gradcheck.py``. What is tested here is that the
forward pass computes the right numbers, that ``unbroadcast`` reduces to the
shape it was asked for, and the handful of choices the ops make at points where
a finite difference has no opinion.
"""

import numpy as np
import pytest

from numgrad import Tensor, exp, log, relu, tanh, unbroadcast

# The three broadcasts a batch of logits produces, as (grad shape, input shape).
BROADCAST_CASES = [
    ((32, 10), (10,)),
    ((32, 10), (32, 1)),
    ((32, 10), (1, 10)),
    ((32, 10), (32, 10)),
    ((32, 10), ()),
]


@pytest.mark.parametrize("grad_shape, input_shape", BROADCAST_CASES)
def test_unbroadcast_returns_the_input_shape(grad_shape, input_shape):
    grad = np.ones(grad_shape)

    assert unbroadcast(grad, input_shape).shape == input_shape


def test_unbroadcast_sums_the_prepended_axis():
    """(32, 10) + (10,): each entry of the bias was copied down 32 rows."""
    grad = np.ones((32, 10))

    np.testing.assert_allclose(unbroadcast(grad, (10,)), np.full(10, 32.0))


def test_unbroadcast_sums_the_stretched_axis():
    """(32, 10) * (32, 1): each entry of the column was copied across 10 slots."""
    grad = np.ones((32, 10))

    np.testing.assert_allclose(unbroadcast(grad, (32, 1)), np.full((32, 1), 10.0))


def test_unbroadcast_keeps_the_length_one_axis():
    """A stretched axis collapses back to 1 rather than disappearing."""
    grad = np.ones((32, 10))

    result = unbroadcast(grad, (1, 10))

    assert result.shape == (1, 10)
    np.testing.assert_allclose(result, np.full((1, 10), 32.0))


def test_unbroadcast_sums_the_right_axis():
    """Uneven values, so summing the wrong axis gives a different answer.

    With a grad of all ones every axis sums to the same constant and an
    unbroadcast that reduced the wrong axis would still look correct.
    """
    grad = np.arange(6.0).reshape((2, 3))

    np.testing.assert_allclose(unbroadcast(grad, (1, 3)), [[3.0, 5.0, 7.0]])
    np.testing.assert_allclose(unbroadcast(grad, (2, 1)), [[3.0], [12.0]])


def test_unbroadcast_of_a_matching_shape_is_unchanged():
    grad = np.arange(6.0).reshape((2, 3))

    np.testing.assert_array_equal(unbroadcast(grad, (2, 3)), grad)


def test_forward_values():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[10.0, 20.0], [30.0, 40.0]])

    np.testing.assert_allclose((x - y).data, [[-9.0, -18.0], [-27.0, -36.0]])
    np.testing.assert_allclose((-x).data, [[-1.0, -2.0], [-3.0, -4.0]])
    np.testing.assert_allclose((y / x).data, [[10.0, 10.0], [10.0, 10.0]])
    np.testing.assert_allclose((x**2).data, [[1.0, 4.0], [9.0, 16.0]])
    np.testing.assert_allclose(exp(x).data, np.exp(x.data))
    np.testing.assert_allclose(log(x).data, np.log(x.data))
    np.testing.assert_allclose(tanh(x).data, np.tanh(x.data))


def test_relu_forward():
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0])

    np.testing.assert_allclose(relu(x).data, [0.0, 0.0, 0.0, 0.5, 2.0])


def test_relu_takes_the_zero_subgradient_at_zero():
    """relu has no derivative at 0, so the backward pass picks one: 0.

    A central difference at 0 measures 1/2, which is why the gradient check for
    relu seeds its inputs away from 0. This test pins the choice so it is a
    decision rather than an accident.
    """
    x = Tensor([-1.0, 0.0, 1.0])

    out = relu(x)
    out.backward(np.ones_like(out.data))

    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])


def test_scalars_are_wrapped_on_both_sides():
    x = Tensor([2.0, 4.0])

    np.testing.assert_allclose((1.0 - x).data, [-1.0, -3.0])
    np.testing.assert_allclose((8.0 / x).data, [4.0, 2.0])


def test_reductions_shapes():
    x = Tensor(np.arange(12.0).reshape((4, 3)))

    assert x.sum().shape == ()
    assert x.sum(axis=0).shape == (3,)
    assert x.sum(axis=1, keepdims=True).shape == (4, 1)
    assert x.max(axis=0).shape == (3,)
    assert x.max(axis=1, keepdims=True).shape == (4, 1)

    np.testing.assert_allclose(x.sum(axis=0).data, x.data.sum(axis=0))
    np.testing.assert_allclose(x.max(axis=1).data, x.data.max(axis=1))


def test_max_sends_the_gradient_to_the_winner_only():
    x = Tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]])

    out = x.max(axis=1)
    out.backward(np.ones_like(out.data))

    np.testing.assert_allclose(x.grad, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_reshape_forward():
    x = Tensor(np.arange(6.0))

    assert x.reshape((2, 3)).shape == (2, 3)
    np.testing.assert_allclose(
        x.reshape((2, 3)).data, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    )


def test_getitem_forward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    np.testing.assert_allclose(x[1].data, [3.0, 4.0])
    np.testing.assert_allclose(x[0:2].data, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(x[np.array([0, 2]), np.array([1, 0])].data, [2.0, 5.0])


def test_getitem_accumulates_a_repeated_index():
    """Row 0 is selected twice, so it takes two contributions, not one.

    ``a.grad[index] += ...`` would silently keep only the last write here, which
    is the bug np.add.at exists to avoid.
    """
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    out = x[np.array([0, 0, 1])]
    out.backward(np.ones_like(out.data))

    np.testing.assert_allclose(x.grad, [[2.0, 2.0], [1.0, 1.0]])


def test_pow_rejects_a_tensor_exponent():
    x = Tensor([1.0, 2.0])

    with pytest.raises(TypeError):
        x ** Tensor([2.0])


def test_ops_do_not_mutate_their_inputs():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0, 6.0], [7.0, 8.0]])
    before_x = x.data.copy()
    before_y = y.data.copy()

    out = tanh(x / y - relu(x) * 2.0)
    out.backward(np.ones_like(out.data))

    np.testing.assert_array_equal(x.data, before_x)
    np.testing.assert_array_equal(y.data, before_y)


def test_everything_stays_float64():
    x = Tensor([[1, 2], [3, 4]])

    for out in [-x, x - 1, x / 2, x**2, exp(x), log(x), relu(x), tanh(x)]:
        assert out.data.dtype == np.float64
        assert out.grad.dtype == np.float64

    assert x.sum(axis=0).data.dtype == np.float64
    assert x.max().data.dtype == np.float64
    assert x[0].data.dtype == np.float64
