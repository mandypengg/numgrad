"""Neural network building blocks.

Layers hold parameter tensors and are callable on a ``Tensor``. A layer exposes
its parameters so an optimizer can collect them; it does not own the update
step.

Planned: Module base, Linear, ReLU, Tanh, Sequential, He/Xavier initializers,
mse_loss, cross_entropy_loss.

Parameters are initialized as ``float64``. Losses are built from the ops in
``numgrad.ops`` so they are differentiated by the same machinery as everything
else, rather than carrying hand-written gradients of their own.

Not implemented yet.
"""
