"""The Tensor type and the backward pass.

A ``Tensor`` wraps a ``float64`` NumPy array and records how it was produced.
Each tensor built by an op keeps a reference to its inputs and to the local
backward function for that op; together these form the tape.

``backward()`` walks the tape in reverse topological order, seeding the output
gradient with ones and calling each node's backward function. Every gradient
write is an accumulation::

    node.grad += contribution

never an assignment, so a tensor used at several points in the graph sums the
contribution from each use. ``grad`` is allocated as a zero array of the
tensor's shape at construction time, which means the accumulation site never
has to special-case an unset gradient.

``zero_grad()`` is the only place a gradient is reset, and it zeros in place.
"""

import numpy as np


class Tensor:
    """A float64 array plus the record of how it was computed.

    Parameters
    ----------
    data:
        Anything ``np.asarray`` accepts. It is cast to ``float64``; there is no
        dtype argument, because finite-difference gradient checks need the
        precision.
    _prev:
        The tensors this one was computed from. Leaves pass nothing.
    _op:
        Short label naming the op that produced this tensor, used by ``__repr__``
        when reading a graph back.
    """

    def __init__(self, data, _prev=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)

        # Allocated up front so every accumulation site can use += unconditionally.
        self.grad = np.zeros_like(self.data)

        # Set membership is by object identity, which is what we want: two
        # distinct tensors holding equal values are still distinct graph nodes.
        self._prev = set(_prev)
        self._op = _op

        # Leaves have nothing to propagate, so the default is a no-op.
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        """Transpose, as a tracked op so gradients flow through it."""
        from numgrad import ops

        return ops.transpose(self)

    def sum(self, axis=None, keepdims=False):
        """Sum, as a tracked op so gradients flow through it."""
        from numgrad import ops

        return ops.sum(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """Maximum, as a tracked op so gradients flow through it."""
        from numgrad import ops

        return ops.max(self, axis=axis, keepdims=keepdims)

    def reshape(self, shape):
        """Reshape, as a tracked op so gradients flow through it."""
        from numgrad import ops

        return ops.reshape(self, shape)

    def zero_grad(self):
        """Reset this tensor's gradient to zeros, in place.

        In place rather than rebinding, so that anything already holding a
        reference to ``.grad`` sees the reset. This is the only place a gradient
        is cleared.
        """
        self.grad[...] = 0.0

    def backward(self):
        """Accumulate gradients of this tensor into every tensor it depends on.

        The traversal is an explicit stack-based DFS rather than a recursive one:
        a graph thousands of nodes deep is normal here, and recursion would hit
        Python's stack limit.
        """
        topo = []
        visited = set()

        # Each stack entry pairs a node with a flag saying whether its parents
        # have already been pushed. Popping an entry with the flag set means
        # every parent is finished, so the node can be appended.
        stack = [(self, False)]
        while stack:
            node, parents_pushed = stack.pop()

            if parents_pushed:
                topo.append(node)
                continue

            if node in visited:
                continue
            visited.add(node)

            # Re-push the node underneath its parents so it is appended last.
            stack.append((node, True))
            for parent in node._prev:
                if parent not in visited:
                    stack.append((parent, False))

        # topo lists parents before children, so the reverse order hands each
        # node its full incoming gradient before it propagates.
        self.grad += np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        # Imported here rather than at module scope because ops.py imports
        # Tensor from this module.
        from numgrad import ops

        return ops.add(self, other)

    def __radd__(self, other):
        from numgrad import ops

        return ops.add(other, self)

    def __sub__(self, other):
        from numgrad import ops

        return ops.sub(self, other)

    def __rsub__(self, other):
        from numgrad import ops

        return ops.sub(other, self)

    def __neg__(self):
        from numgrad import ops

        return ops.neg(self)

    def __mul__(self, other):
        from numgrad import ops

        return ops.mul(self, other)

    def __rmul__(self, other):
        from numgrad import ops

        return ops.mul(other, self)

    def __truediv__(self, other):
        from numgrad import ops

        return ops.truediv(self, other)

    def __rtruediv__(self, other):
        from numgrad import ops

        return ops.truediv(other, self)

    def __pow__(self, exponent):
        from numgrad import ops

        return ops.pow(self, exponent)

    def __getitem__(self, index):
        from numgrad import ops

        return ops.getitem(self, index)

    def __matmul__(self, other):
        from numgrad import ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other):
        from numgrad import ops

        return ops.matmul(other, self)

    def __repr__(self):
        if self._op:
            return f"Tensor(shape={self.shape}, op={self._op!r})"
        return f"Tensor(shape={self.shape})"
