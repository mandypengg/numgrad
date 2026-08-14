"""Neural network building blocks.

Layers hold parameter tensors and are callable on a ``Tensor``. A layer exposes
its parameters so an optimizer can collect them; it does not own the update
step.

Planned: Xavier initializer, mse_loss.

Parameters are initialized as ``float64``. Losses are built from the ops in
``numgrad.ops`` so they are differentiated by the same machinery as everything
else, rather than carrying hand-written gradients of their own.

Implemented so far: Module, Linear, ReLU, Tanh, Sequential,
softmax_cross_entropy.
"""

import numpy as np

from numgrad import ops
from numgrad.tensor import Tensor


class Module:
    """Base class for anything callable that may hold parameters.

    Subclasses implement ``forward`` and, if they own parameter tensors,
    override ``parameters``. The base class deliberately does no bookkeeping of
    its own: there is no registry that discovers parameters by inspecting
    attributes, because a reader tracing which tensors an optimizer updates
    should be able to read the answer off the ``parameters`` method rather than
    off an attribute hook.
    """

    def parameters(self):
        """The tensors an optimizer should update. Empty unless overridden."""
        return []

    def zero_grad(self):
        """Clear the gradient of every parameter this module owns."""
        for param in self.parameters():
            param.zero_grad()

    def forward(self, x):
        raise NotImplementedError(f"{type(self).__name__} does not implement forward()")

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """An affine map ``x @ weight + bias``.

    Parameters
    ----------
    fan_in:
        Number of input features. Rows of ``weight``.
    fan_out:
        Number of output features. Columns of ``weight``.
    rng:
        Optional ``np.random.Generator``, so a test can fix the initialization.

    Weights use He initialization: ``randn(fan_in, fan_out) * sqrt(2 / fan_in)``.
    The scale is what keeps activations from shrinking or blowing up layer after
    layer. A unit-variance input times a weight column of ``fan_in`` independent
    entries of variance ``s**2`` gives a pre-activation of variance
    ``fan_in * s**2``, and a ReLU then zeroes about half the distribution, which
    halves that variance again. Setting ``s**2 = 2 / fan_in`` cancels both
    factors and leaves the output at unit variance.

    The bias starts at zeros. It has no fan-in to compensate for, and a nonzero
    bias would only break the symmetry that the random weights have already
    broken.
    """

    def __init__(self, fan_in, fan_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        scale = np.sqrt(2.0 / fan_in)
        self.weight = Tensor(rng.standard_normal((fan_in, fan_out)) * scale)

        # Shape (fan_out,) rather than (1, fan_out): it broadcasts across the
        # batch either way, and the backward pass sums the broadcast axis away.
        self.bias = Tensor(np.zeros(fan_out))

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        return ops.matmul(x, self.weight) + self.bias


class ReLU(Module):
    """Elementwise ``max(x, 0)``, with no parameters."""

    def forward(self, x):
        return ops.relu(x)


class Tanh(Module):
    """Elementwise hyperbolic tangent, with no parameters."""

    def forward(self, x):
        return ops.tanh(x)


class Sequential(Module):
    """Chain of modules, applied in the order given."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def parameters(self):
        # Written as a loop rather than a comprehension over comprehensions, so
        # the order is plainly the order the layers were given in. Optimizers
        # rely on that order only for reporting, but a reader should not have to
        # take it on faith.
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def softmax_cross_entropy(logits, labels):
    """Mean cross entropy over a batch, taken directly from raw logits.

    Parameters
    ----------
    logits:
        ``(batch, classes)`` tensor of unnormalized scores.
    labels:
        ``(batch,)`` array of integer class indices, one per row. These are
        plain integers, not a Tensor: they index the graph rather than flow
        through it, so nothing is differentiated with respect to them.

    Returns
    -------
    Tensor
        Scalar loss, averaged over the batch.

    The softmax is never formed on its own. Written out, the loss for one row is

        -log(exp(z_y) / sum_j exp(z_j)) = log(sum_j exp(z_j)) - z_y

    and computing ``exp`` of a raw logit is what overflows: ``exp(1e3)`` is
    ``inf`` in float64, and the ``inf / inf`` that follows is a nan that then
    poisons the whole backward pass. Subtracting the row maximum first fixes
    that, because adding any constant c to every logit in a row leaves the loss
    unchanged (the c cancels between the two terms above), so we are free to
    pick c = -max and cap the largest exponent at ``exp(0) = 1``. Every other
    exponent is then in (0, 1], which can only underflow to 0, and an underflowed
    term is a probability that really was negligible.
    """
    logits = ops.as_tensor(logits)
    labels = np.asarray(labels)

    if logits.data.ndim != 2:
        raise ValueError(
            f"softmax_cross_entropy expects 2-D logits, got shape {logits.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"expected one label per row: logits shape {logits.shape}, "
            f"labels shape {labels.shape}"
        )

    batch = logits.shape[0]

    # The row max stays on the tape rather than being peeled off as a constant.
    # It costs nothing: the gradient flowing back into it is the negated row sum
    # of (softmax - one hot), and that sum is zero, which is the same shift
    # invariance the trick relies on in the forward pass.
    shifted = logits - logits.max(axis=1, keepdims=True)

    # log(sum(exp)) of the shifted row. Held keepdims so it broadcasts back
    # across the classes on the next line.
    row_sums = ops.exp(shifted).sum(axis=1, keepdims=True)
    log_probs = shifted - ops.log(row_sums)

    # One log-probability per row: the one belonging to that row's true class.
    rows = np.arange(batch)
    picked = log_probs[rows, labels]

    return -picked.sum() / batch
