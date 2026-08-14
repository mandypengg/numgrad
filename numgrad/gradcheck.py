"""Finite-difference gradient checking.

The analytic gradient produced by an op's backward pass is compared against a
central difference of the forward pass::

    (f(x + h) - f(x - h)) / (2h)

taken one scalar entry at a time, for every entry of every parameter. Central
differences rather than forward ones because the error term is O(h^2) instead
of O(h), which is what makes a tolerance this tight usable at all. This is also
why the library is float64 throughout: at float32 the roundoff in the numerator
would swamp the signal.

The loss being differentiated is ``f(*params).data.sum()``. Summing gives one
well-defined scalar to difference, and it is the same quantity ``backward()``
computes gradients of, since ``backward()`` seeds the output with ones.

Agreement is measured as a relative error::

    rel = |num - ana| / max(1e-8, |num| + |ana|)

The floor in the denominator keeps the ratio finite when both gradients are
zero or nearly so, which would otherwise divide a tiny difference by a tiny
sum and report a large error for two numbers that agree.
"""

import numpy as np

from numgrad.tensor import Tensor


def check_grads(f, params, h=1e-5, tol=1e-6):
    """Check ``f``'s analytic gradients against central differences.

    Parameters
    ----------
    f:
        Callable taking the parameters positionally and returning a single
        ``Tensor``. It is called many times, once per scalar entry per side of
        the difference, so it must not have side effects.
    params:
        The inputs to differentiate with respect to. ``Tensor`` instances are
        used as they are; anything else is wrapped in one.
    h:
        Step size for the difference. Too large and the O(h^2) truncation term
        shows up; too small and cancellation in ``f(x + h) - f(x - h)`` does.
    tol:
        Largest relative error accepted for a single entry.

    Raises
    ------
    AssertionError
        On the first entry whose relative error exceeds ``tol``, naming the op
        that produced the output, the offending index, and both gradients.
    """
    params = [p if isinstance(p, Tensor) else Tensor(p) for p in params]

    # Cleared first so a caller who already ran a backward pass on these
    # tensors does not have those gradients added to the ones measured here.
    for param in params:
        param.zero_grad()

    out = f(*params)

    # The op that produced the output, which is where a failure points first.
    # A composite graph reports its last op; leaves report nothing.
    op_name = out._op or "<leaf>"

    out.backward()

    # Copied because the numeric pass below leaves each param's .grad in place,
    # and a caller's later use of it should not see values captured mid-check.
    analytic = [param.grad.copy() for param in params]

    for position, param in enumerate(params):
        for index in np.ndindex(param.data.shape):
            original = param.data[index]

            # Perturbed in place and restored, so f sees the same tensor
            # objects on every call and no rebuilding is needed.
            param.data[index] = original + h
            plus = float(f(*params).data.sum())

            param.data[index] = original - h
            minus = float(f(*params).data.sum())

            param.data[index] = original

            numeric = (plus - minus) / (2.0 * h)
            analytic_value = float(analytic[position][index])

            rel = abs(numeric - analytic_value) / max(
                1e-8, abs(numeric) + abs(analytic_value)
            )

            if rel > tol:
                raise AssertionError(
                    f"gradient check failed for op {op_name!r}\n"
                    f"  param {position} of {len(params)}, "
                    f"shape {param.shape}, index {index}\n"
                    f"  numeric  = {numeric!r}\n"
                    f"  analytic = {analytic_value!r}\n"
                    f"  relative error = {rel:.6e} "
                    f"(tol = {tol:.6e}, h = {h:.6e})"
                )
