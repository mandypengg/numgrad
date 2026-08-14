"""Primitive operations: forward computation and local gradients.

One op per function, written out longhand. Each op computes its forward value
and returns a ``Tensor`` carrying a backward function that accumulates into its
inputs' gradients.

Where an op broadcasts, the backward pass has to undo that broadcast: sum the
incoming gradient over the axes that were expanded, then reshape to the input's
original shape. That is the same three lines in every elementwise op, and
getting one of them subtly wrong is hard to see, so it lives in ``unbroadcast``
below and every elementwise backward routes through it. Read that function once
and the rest of the file is chain rule and nothing else.

Planned: mean, sigmoid, softmax.

Every op added here needs a finite-difference gradient test in ``tests/`` in the
same commit.

Implemented so far: add, sub, mul, truediv, pow, neg, matmul, exp, log, relu,
tanh, sum, max, reshape, transpose, getitem.
"""

import numpy as np

from numgrad.tensor import Tensor


def as_tensor(x):
    """Wrap a scalar or array operand so ops only ever see Tensors."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def unbroadcast(grad, shape):
    """Reduce ``grad`` back to ``shape``, undoing a forward broadcast.

    NumPy broadcasts in two steps, and this reverses both of them. It prepends
    axes to the smaller operand until the ranks match, and it stretches any axis
    of length 1 to the other operand's length. Both steps copy one input value
    into many output positions, so the gradient of that input value is the sum
    of the gradients at every position it was copied to.
    """
    # Prepended axes: the input has no corresponding axis at all, so sum them
    # away entirely.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Stretched axes: the input has the axis but with length 1, so collapse the
    # axis back to length 1 rather than removing it.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


def add(a, b):
    """Elementwise sum, with NumPy broadcasting."""
    a = as_tensor(a)
    b = as_tensor(b)

    out = Tensor(a.data + b.data, _prev=(a, b), _op="add")

    def _backward():
        # d(a + b)/da is 1, so the incoming gradient passes straight through
        # and the only work is undoing the broadcast.
        a.grad += unbroadcast(out.grad, a.data.shape)
        b.grad += unbroadcast(out.grad, b.data.shape)

    out._backward = _backward
    return out


def sub(a, b):
    """Elementwise difference, with NumPy broadcasting."""
    a = as_tensor(a)
    b = as_tensor(b)

    out = Tensor(a.data - b.data, _prev=(a, b), _op="sub")

    def _backward():
        # d(a - b)/da is 1 and d(a - b)/db is -1.
        a.grad += unbroadcast(out.grad, a.data.shape)
        b.grad += unbroadcast(-out.grad, b.data.shape)

    out._backward = _backward
    return out


def mul(a, b):
    """Elementwise product, with NumPy broadcasting."""
    a = as_tensor(a)
    b = as_tensor(b)

    out = Tensor(a.data * b.data, _prev=(a, b), _op="mul")

    def _backward():
        # d(a * b)/da is b, so scale the incoming gradient by the other operand
        # before undoing the broadcast.
        a.grad += unbroadcast(out.grad * b.data, a.data.shape)
        b.grad += unbroadcast(out.grad * a.data, b.data.shape)

    out._backward = _backward
    return out


def truediv(a, b):
    """Elementwise division, with NumPy broadcasting."""
    a = as_tensor(a)
    b = as_tensor(b)

    out = Tensor(a.data / b.data, _prev=(a, b), _op="truediv")

    def _backward():
        # d(a / b)/da is 1 / b.
        a.grad += unbroadcast(out.grad / b.data, a.data.shape)
        # d(a / b)/db is -a / b**2.
        b.grad += unbroadcast(-out.grad * a.data / (b.data * b.data), b.data.shape)

    out._backward = _backward
    return out


def pow(a, exponent):
    """Raise a tensor to a constant power.

    The exponent is a plain number rather than a Tensor. A tensor exponent needs
    ``log(a)`` in its backward pass, which is undefined for the negative bases
    this op otherwise handles fine, and nothing here needs one.
    """
    a = as_tensor(a)

    if isinstance(exponent, Tensor):
        raise TypeError("pow expects a constant exponent, not a Tensor")

    out = Tensor(a.data**exponent, _prev=(a,), _op="pow")

    def _backward():
        # d(a ** p)/da = p * a ** (p - 1). No broadcast to undo: the exponent is
        # a scalar, so the output has a's shape.
        a.grad += out.grad * exponent * a.data ** (exponent - 1)

    out._backward = _backward
    return out


def neg(a):
    """Elementwise negation."""
    a = as_tensor(a)

    out = Tensor(-a.data, _prev=(a,), _op="neg")

    def _backward():
        a.grad += -out.grad

    out._backward = _backward
    return out


def matmul(a, b):
    """Matrix product of two 2-D tensors.

    Restricted to 2-D on both sides. Stacked and 1-D cases have a backward pass
    with enough shape bookkeeping to obscure the rule being applied, so they are
    left out until something needs them.
    """
    a = as_tensor(a)
    b = as_tensor(b)

    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(
            f"matmul expects two 2-D tensors, got shapes {a.shape} and {b.shape}"
        )

    out = Tensor(a.data @ b.data, _prev=(a, b), _op="matmul")

    def _backward():
        # For C = A @ B, dL/dA = dL/dC @ B.T and dL/dB = A.T @ dL/dC. Both
        # shapes come out right by construction, so there is no broadcast to
        # undo here.
        a.grad += out.grad @ b.data.T
        b.grad += a.data.T @ out.grad

    out._backward = _backward
    return out


def exp(a):
    """Elementwise exponential."""
    a = as_tensor(a)

    out = Tensor(np.exp(a.data), _prev=(a,), _op="exp")

    def _backward():
        # d(exp(a))/da is exp(a), which is the forward value already computed.
        a.grad += out.grad * out.data

    out._backward = _backward
    return out


def log(a):
    """Elementwise natural logarithm."""
    a = as_tensor(a)

    out = Tensor(np.log(a.data), _prev=(a,), _op="log")

    def _backward():
        a.grad += out.grad / a.data

    out._backward = _backward
    return out


def relu(a):
    """Elementwise ``max(a, 0)``.

    Not differentiable at exactly 0. The backward pass takes the subgradient 0
    there, which is the usual choice; a finite-difference check straddling 0
    would disagree with it, so gradient tests seed inputs away from 0.
    """
    a = as_tensor(a)

    out = Tensor(np.maximum(a.data, 0.0), _prev=(a,), _op="relu")

    def _backward():
        # Gradient passes through where the input was positive and is stopped
        # everywhere else, including at exactly 0.
        a.grad += out.grad * (a.data > 0.0)

    out._backward = _backward
    return out


def tanh(a):
    """Elementwise hyperbolic tangent."""
    a = as_tensor(a)

    out = Tensor(np.tanh(a.data), _prev=(a,), _op="tanh")

    def _backward():
        # d(tanh(a))/da = 1 - tanh(a)**2, again reusing the forward value.
        a.grad += out.grad * (1.0 - out.data * out.data)

    out._backward = _backward
    return out


def sum(a, axis=None, keepdims=False):
    """Sum over ``axis``, or over every entry when ``axis`` is None."""
    a = as_tensor(a)

    out = Tensor(np.sum(a.data, axis=axis, keepdims=keepdims), _prev=(a,), _op="sum")

    def _backward():
        grad = out.grad
        # Summing without keepdims drops the reduced axis from the output, so
        # put it back before broadcasting the gradient over it.
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis)
        # Every input entry contributed exactly once to the sum it landed in, so
        # each one receives that sum's gradient unscaled.
        a.grad += np.broadcast_to(grad, a.data.shape)

    out._backward = _backward
    return out


def max(a, axis=None, keepdims=False):
    """Maximum over ``axis``, or over every entry when ``axis`` is None."""
    a = as_tensor(a)

    out = Tensor(np.max(a.data, axis=axis, keepdims=keepdims), _prev=(a,), _op="max")

    def _backward():
        # Compared against the keepdims maximum so the comparison broadcasts
        # back across the reduced axis regardless of how out was shaped.
        winners = a.data == np.max(a.data, axis=axis, keepdims=True)

        grad = out.grad
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis)

        # Only the entry that won the maximum affects the output, so it takes
        # the whole gradient and the rest take none. On an exact tie every
        # winner takes the full gradient, which no finite difference agrees
        # with; random float64 inputs do not tie.
        a.grad += winners * grad

    out._backward = _backward
    return out


def reshape(a, shape):
    """View the same entries under a new shape."""
    a = as_tensor(a)

    out = Tensor(a.data.reshape(shape), _prev=(a,), _op="reshape")

    def _backward():
        # Reshaping moves entries around without combining them, so the
        # gradient just moves back.
        a.grad += out.grad.reshape(a.data.shape)

    out._backward = _backward
    return out


def transpose(a):
    """Reverse the axes of a tensor."""
    a = as_tensor(a)

    out = Tensor(a.data.T, _prev=(a,), _op="transpose")

    def _backward():
        # Transposing is its own inverse, so the gradient is transposed back.
        a.grad += out.grad.T

    out._backward = _backward
    return out


def getitem(a, index):
    """Select entries with NumPy indexing, keeping the selection on the tape.

    This is what pulls the correct class's logit out of each row of a batch:
    ``logits[np.arange(batch), labels)]``.
    """
    a = as_tensor(a)

    out = Tensor(a.data[index], _prev=(a,), _op="getitem")

    def _backward():
        # An index array can name the same entry more than once, and
        # `a.grad[index] += ...` would keep only the last of those writes
        # instead of summing them. np.add.at accumulates every one.
        np.add.at(a.grad, index, out.grad)

    out._backward = _backward
    return out
