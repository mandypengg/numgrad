"""Neural network building blocks.

Layers hold parameter tensors and are callable on a ``Tensor``. A layer exposes
its parameters so an optimizer can collect them; it does not own the update
step.

Planned: Module base, Linear, ReLU, Tanh, Sequential, He/Xavier initializers,
mse_loss.

Parameters are initialized as ``float64``. Losses are built from the ops in
``numgrad.ops`` so they are differentiated by the same machinery as everything
else, rather than carrying hand-written gradients of their own.

Implemented so far: softmax_cross_entropy.
"""

import numpy as np

from numgrad import ops


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
