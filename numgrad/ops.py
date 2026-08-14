"""Primitive operations: forward computation and local gradients.

One op per function, written out longhand. Each op computes its forward value
and returns a ``Tensor`` carrying a backward function that accumulates into its
inputs' gradients.

Where an op broadcasts, the backward pass has to undo that broadcast: sum the
incoming gradient over the axes that were expanded, then reshape to the input's
original shape. This is written explicitly in each op rather than factored into
a shared helper, because it is the part readers most often need to trace.

Planned: add, mul, matmul, sub, neg, div, pow, sum, mean, reshape, transpose,
exp, log, relu, tanh, sigmoid, max, softmax.

Every op added here needs a finite-difference gradient test in ``tests/`` in the
same commit.

Implemented so far: add, mul, matmul, transpose.
"""

import numpy as np

from numgrad.tensor import Tensor


def as_tensor(x):
    """Wrap a scalar or array operand so ops only ever see Tensors."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def add(a, b):
    """Elementwise sum, with NumPy broadcasting."""
    a = as_tensor(a)
    b = as_tensor(b)

    out = Tensor(a.data + b.data, _prev=(a, b), _op="add")

    def _backward():
        # d(a + b)/da is 1, so the incoming gradient passes straight through
        # and the only work is undoing the broadcast.
        grad_a = out.grad
        # Broadcasting prepends axes to the smaller operand; sum those away.
        while grad_a.ndim > a.data.ndim:
            grad_a = grad_a.sum(axis=0)
        # Any axis where a had length 1 was stretched; sum it back to length 1.
        for axis, size in enumerate(a.data.shape):
            if size == 1 and grad_a.shape[axis] != 1:
                grad_a = grad_a.sum(axis=axis, keepdims=True)
        a.grad += grad_a.reshape(a.data.shape)

        grad_b = out.grad
        while grad_b.ndim > b.data.ndim:
            grad_b = grad_b.sum(axis=0)
        for axis, size in enumerate(b.data.shape):
            if size == 1 and grad_b.shape[axis] != 1:
                grad_b = grad_b.sum(axis=axis, keepdims=True)
        b.grad += grad_b.reshape(b.data.shape)

    out._backward = _backward
    return out


def mul(a, b):
    """Elementwise product, with NumPy broadcasting."""
    a = as_tensor(a)
    b = as_tensor(b)

    out = Tensor(a.data * b.data, _prev=(a, b), _op="mul")

    def _backward():
        # d(a * b)/da is b, so scale the incoming gradient by the other operand
        # and then undo the broadcast exactly as in add.
        grad_a = out.grad * b.data
        while grad_a.ndim > a.data.ndim:
            grad_a = grad_a.sum(axis=0)
        for axis, size in enumerate(a.data.shape):
            if size == 1 and grad_a.shape[axis] != 1:
                grad_a = grad_a.sum(axis=axis, keepdims=True)
        a.grad += grad_a.reshape(a.data.shape)

        grad_b = out.grad * a.data
        while grad_b.ndim > b.data.ndim:
            grad_b = grad_b.sum(axis=0)
        for axis, size in enumerate(b.data.shape):
            if size == 1 and grad_b.shape[axis] != 1:
                grad_b = grad_b.sum(axis=axis, keepdims=True)
        b.grad += grad_b.reshape(b.data.shape)

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


def transpose(a):
    """Reverse the axes of a tensor."""
    a = as_tensor(a)

    out = Tensor(a.data.T, _prev=(a,), _op="transpose")

    def _backward():
        # Transposing is its own inverse, so the gradient is transposed back.
        a.grad += out.grad.T

    out._backward = _backward
    return out
