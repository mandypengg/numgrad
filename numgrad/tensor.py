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

The walk is seeded with the gradient of the quantity being differentiated with
respect to the output. That defaults to 1.0, which only means something for a
scalar, so a non-scalar output has to say what it wants. Passing a seed of ones
differentiates the output's sum; any other seed gives the vector-Jacobian
product, which is the operation reverse mode actually performs.

``no_grad`` switches recording off for a block, so a forward pass whose values
are the whole point does not build a tape that is immediately thrown away.
"""

import numpy as np

# Ops record onto the tape unless a ``no_grad`` block switches this off. It is
# read in exactly one place, ``Tensor.__init__``, because every op builds its
# output by calling it.
_recording = True


class no_grad:
    """Context manager that stops ops from recording onto the tape.

    Inside the block, every tensor an op returns is built as a leaf: no inputs,
    no op label, and nothing for ``backward()`` to walk::

        with no_grad():
            logits = model(x)

    The forward values are exactly what they would be otherwise. Only the
    bookkeeping is skipped, which is the point at evaluation time, where the
    graph would be built and then dropped without a backward pass.

    The previous state is saved and restored rather than reset to True, so a
    nested block does not switch recording back on when the inner one exits.
    """

    def __enter__(self):
        global _recording

        self._previous = _recording
        _recording = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _recording

        _recording = self._previous
        # False, so an exception raised inside the block still propagates.
        return False


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
        # Under no_grad the tape is not recorded and the tensor is built as a
        # leaf instead. An op still assigns _backward below, but backward()
        # never calls it, because a node with no inputs has nowhere to send a
        # gradient.
        if _recording:
            self._prev = set(_prev)
            self._op = _op
        else:
            self._prev = set()
            self._op = ""

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

    def relu(self):
        """Elementwise max(self, 0), as a tracked op so gradients flow through it."""
        from numgrad import ops

        return ops.relu(self)

    def detach(self):
        """A leaf tensor holding this one's values, cut off from the tape.

        The result records no inputs and no op, so a backward pass that reaches
        it stops there. Use it to feed a computed value forward without letting
        a gradient flow back through how it was computed.

        The data is copied. ``np.asarray`` hands back a float64 array unchanged
        rather than copying it, so without this the detached tensor would share
        the buffer and an optimizer's in place update would show up in both.
        """
        return Tensor(self.data.copy())

    def zero_grad(self):
        """Reset this tensor's gradient to zeros, in place.

        In place rather than rebinding, so that anything already holding a
        reference to ``.grad`` sees the reset. This is the only place a gradient
        is cleared.
        """
        self.grad[...] = 0.0

    def backward(self, grad=None):
        """Accumulate gradients of this tensor into every tensor it depends on.

        Parameters
        ----------
        grad:
            The gradient of the quantity being differentiated with respect to
            this tensor, which seeds the reverse walk. Defaults to 1.0, which
            only means something when this tensor holds a single number, so a
            larger output has to pass a seed explicitly. ``np.ones_like`` is the
            seed that differentiates the output's sum; any other seed v gives
            the vector-Jacobian product v @ J, one row of the Jacobian at a
            time, which is the operation reverse mode is built to perform.

        The traversal is an explicit stack-based DFS rather than a recursive one:
        a graph thousands of nodes deep is normal here, and recursion would hit
        Python's stack limit.

        The seed accumulates like every other gradient, so a second call on the
        same graph does not repeat the first result, it compounds it: the second
        walk starts from a seed of 2. A training loop builds a fresh graph every
        step, so this only comes up on a graph deliberately held onto, and the
        fix there is to zero every node rather than only the parameters.
        """
        if grad is None:
            # A default seed of 1.0 says "differentiate this number", which is
            # not a question a tensor of many numbers has a single answer to.
            if self.data.size != 1:
                raise ValueError(
                    f"backward() on a tensor of shape {self.shape} needs an "
                    f"explicit seed, because a default of 1.0 only means "
                    f"something for a single number. Pass "
                    f"np.ones_like(t.data) to differentiate its sum."
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=np.float64)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"backward() seed has shape {seed.shape}, but it seeds a "
                    f"tensor of shape {self.shape}"
                )

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
        self.grad += seed
        for node in reversed(topo):
            # A node with no recorded inputs is a leaf and has nowhere to send a
            # gradient. Skipping it is also what stops a graph built under
            # no_grad from propagating: every node in one is a leaf.
            if node._prev:
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
