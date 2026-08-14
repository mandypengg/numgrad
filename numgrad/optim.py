"""Parameter update rules.

An optimizer takes a list of parameter tensors and a learning rate, and exposes
``step()`` and ``zero_grad()``. ``step()`` reads ``param.grad`` and updates
``param.data`` in place; it never writes to ``grad``.

Clearing gradients is always an explicit ``zero_grad()`` call between iterations
— the accumulation in the backward pass has no idea where one training step
ends and the next begins.

Both optimizers here keep one state buffer per parameter, allocated up front and
updated in place, so the buffer a parameter is paired with never changes
identity across steps.

Implemented so far: SGD, Adam.
"""

import numpy as np


class SGD:
    """Gradient descent, optionally with momentum and weight decay.

    Parameters
    ----------
    params:
        The tensors to update, usually ``model.parameters()``.
    lr:
        Learning rate. Scales the whole update.
    momentum:
        Fraction of the previous velocity carried into this step. 0 gives plain
        gradient descent.
    weight_decay:
        Coefficient of an L2 penalty on the parameters.

    The velocity is the running sum

        v = momentum * v + grad

    and the parameter moves by ``-lr * v``. Written this way, a constant
    gradient drives the velocity to ``grad / (1 - momentum)`` rather than to
    ``grad``, so momentum 0.9 takes steps about ten times longer than plain
    descent at the same learning rate. That matters when tuning: raising
    momentum without lowering lr changes the effective step size.
    """

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # One velocity per parameter, matching shapes, starting at rest.
        self.velocities = [np.zeros_like(param.data) for param in self.params]

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        for param, velocity in zip(self.params, self.velocities):
            # A fresh array rather than an in-place add, because param.grad
            # belongs to the backward pass and step() only ever reads it.
            grad = param.grad + self.weight_decay * param.data

            # In place, so the buffer in self.velocities is the one that carries
            # to the next step rather than a rebound local.
            velocity *= self.momentum
            velocity += grad

            param.data -= self.lr * velocity


class Adam:
    """Adaptive moment estimation, with bias correction.

    Parameters
    ----------
    params:
        The tensors to update.
    lr:
        Learning rate. With the normalization below, this is close to the actual
        distance a parameter moves per step early in training.
    b1, b2:
        Decay rates for the running mean and the running mean of squares.
    eps:
        Added to the denominator so a parameter whose gradient has been zero
        throughout does not divide by zero.

    Adam keeps two running averages per parameter: ``m`` estimates the gradient
    and ``v`` estimates its square. The update divides one by the square root of
    the other, so each parameter moves by roughly ``lr`` in the direction of its
    gradient's sign, regardless of that gradient's magnitude.

    Both averages start at zero, which biases them toward zero for the first
    steps: after one step ``m`` is ``(1 - b1) * grad``, a tenth of the gradient
    at the default b1. Dividing by ``1 - b1 ** t`` undoes exactly that shrinkage
    at every t, and the same for ``v``. Without the correction the first steps
    are badly scaled: ``m`` is short by 10x while ``sqrt(v)`` is short by about
    32x, so the ratio comes out roughly 3x too large rather than too small.
    """

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        # Step count, used only by the bias correction. It starts at 0 and is
        # incremented before the first correction, so the first step divides by
        # 1 - b1 rather than by 0.
        self.t = 0

        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        self.t += 1

        # Computed once per step rather than once per parameter: t is shared, so
        # both corrections are the same for every parameter in this step.
        correction1 = 1.0 - self.b1**self.t
        correction2 = 1.0 - self.b2**self.t

        for param, m, v in zip(self.params, self.m, self.v):
            grad = param.grad

            # Updated in place for the same reason as SGD's velocity.
            m *= self.b1
            m += (1.0 - self.b1) * grad

            v *= self.b2
            v += (1.0 - self.b2) * (grad * grad)

            # The corrected values are locals: the stored buffers stay
            # uncorrected, because the correction is a function of t and would
            # be applied again on top of itself next step.
            m_hat = m / correction1
            v_hat = v / correction2

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
