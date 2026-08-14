"""Parameter update rules.

An optimizer takes a list of parameter tensors and a learning rate, and exposes
``step()`` and ``zero_grad()``. ``step()`` reads ``param.grad`` and updates
``param.data`` in place; it never writes to ``grad``.

Clearing gradients is always an explicit ``zero_grad()`` call between iterations
— the accumulation in the backward pass has no idea where one training step
ends and the next begins.

Planned: SGD, SGD with momentum, Adam.

Not implemented yet.
"""
