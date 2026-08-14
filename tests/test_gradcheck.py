"""Finite-difference gradient checks.

Every op in ``numgrad.ops`` gets a check here, added in the same commit as the
op itself. The pattern for each: build random ``float64`` inputs, run the
forward pass, call ``backward()``, and compare each input's ``.grad`` against a
central difference

    (f(x + h) - f(x - h)) / (2h)

taken one element at a time.

Also covered here: a tensor used twice in the same expression must end up with
the sum of both contributions, not the last one.
"""

import numpy as np

from numgrad import Tensor


def numeric_grad(loss_fn, x, h=1e-6):
    """Central-difference gradient of a scalar loss with respect to array ``x``.

    ``loss_fn`` takes no arguments and reads ``x`` as it stands, so each element
    is perturbed in place and then restored.
    """
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index]

        x[index] = original + h
        plus = loss_fn()

        x[index] = original - h
        minus = loss_fn()

        x[index] = original
        grad[index] = (plus - minus) / (2.0 * h)

    return grad


def analytic_grads(build, arrays):
    """Run ``build`` on fresh tensors, sum the output, and return input grads.

    ``build`` takes the tensors positionally and returns a single tensor. The
    output is summed to a scalar so there is one well-defined loss to
    differentiate, which is what the finite difference approximates too.
    """
    tensors = [Tensor(array) for array in arrays]
    out = build(*tensors)

    # backward() seeds the output gradient with ones, which is exactly the
    # gradient of sum(out) with respect to out.
    out.backward()

    return [tensor.grad for tensor in tensors]


def check(build, arrays, tolerance=1e-6):
    """Compare analytic gradients against central differences for each input."""
    grads = analytic_grads(build, arrays)

    for position, array in enumerate(arrays):

        def loss():
            tensors = [Tensor(other) for other in arrays]
            return float(build(*tensors).data.sum())

        expected = numeric_grad(loss, array)
        np.testing.assert_allclose(
            grads[position], expected, rtol=tolerance, atol=tolerance
        )


def test_gradcheck_add():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))

    check(lambda x, y: x + y, [a, b])


def test_gradcheck_add_broadcast():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 3))
    # Row vector, stretched down the rows of a.
    b = rng.standard_normal((1, 3))

    check(lambda x, y: x + y, [a, b])


def test_gradcheck_mul():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal((4, 3))

    check(lambda x, y: x * y, [a, b])


def test_gradcheck_mul_broadcast():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 3))
    # Column vector, stretched across the columns of a.
    b = rng.standard_normal((4, 1))

    check(lambda x, y: x * y, [a, b])


def test_gradcheck_matmul():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((3, 5))
    b = rng.standard_normal((5, 2))

    check(lambda x, y: x @ y, [a, b])


def test_gradcheck_transpose():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 5))
    b = rng.standard_normal((3, 5))

    # Multiplied by a second input so the transpose is not the whole graph.
    check(lambda x, y: (x * y).T, [a, b])


def test_gradcheck_diamond():
    """x feeds two branches that recombine, so its gradient must sum both."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((4, 3))

    def build(x, w):
        left = x * w
        right = x + w
        return left + right

    check(build, [x, w])


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


def test_repeated_use_deeper_in_the_graph():
    """x reaching the output by three separate routes sums all three."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal((3, 3))

    def build(x):
        squared = x * x
        return squared + x

    check(build, [x])
