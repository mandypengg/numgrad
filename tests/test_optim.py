"""Tests for the update rules in ``numgrad.optim``.

An optimizer has no gradients of its own to check, so these tests come in two
kinds. The mechanical ones step a parameter once or twice and compare against
the update rule worked out by hand, which is where an off-by-one in the bias
correction or a velocity that fails to persist would show. The convergence ones
point each optimizer at a quadratic whose minimum is known in closed form and
require it to get there within a step budget.

The quadratic is deterministic, so every step count below is exact rather than
typical: these tests do not flake, they either pass at these constants or fail.
"""

import numpy as np

from numgrad import SGD, Adam, Tensor

# A bowl with its minimum at MINIMUM, three times steeper along the second axis
# than the first. The asymmetry matters: on a perfectly round bowl every
# gradient points straight at the minimum, and an optimizer that mishandles
# per-coordinate scaling still looks fine.
CURVATURE = np.array([1.0, 3.0])
MINIMUM = np.array([3.0, -2.0])
START = np.array([-4.0, 5.0])


def quadratic(w):
    """``sum(curvature * (w - minimum) ** 2)``, minimized at MINIMUM with value 0."""
    return (Tensor(CURVATURE) * (w - Tensor(MINIMUM)) ** 2).sum()


def steps_to_reach_minimum(optimizer, w, tol, budget=200):
    """Run until every coordinate is within ``tol`` of MINIMUM, or give up.

    Returns the number of steps taken, or None if the budget ran out. The
    gradients are cleared at the top of each iteration rather than the bottom,
    since the backward pass accumulates and last iteration's gradient is still
    sitting there.
    """
    for step in range(budget):
        optimizer.zero_grad()
        quadratic(w).backward()
        optimizer.step()

        if np.abs(w.data - MINIMUM).max() < tol:
            return step + 1

    return None


# Convergence. The tolerance is 1e-4, and it is set by Adam rather than by SGD:
# see test_adam_orbits_the_minimum_rather_than_landing_on_it for why a constant
# learning rate stops buying precision around there. Plain SGD reaches the same
# point in a fraction of the steps and would happily go to 1e-9.
#
# The criterion is first arrival, which is the right one for Adam: it arrives
# and then keeps orbiting, so a check made at a fixed step number would depend
# on where in the orbit that step fell.

TOL = 1e-4


def test_sgd_reaches_the_minimum():
    w = Tensor(START.copy())

    steps = steps_to_reach_minimum(SGD([w], lr=0.2), w, TOL)

    assert steps is not None and steps < 200
    np.testing.assert_allclose(w.data, MINIMUM, atol=TOL)


def test_sgd_with_momentum_reaches_the_minimum():
    """Momentum 0.9 needs roughly ten times the steps plain descent does here.

    That is not a bad learning rate, it is what momentum does on a quadratic.
    Above a fairly low learning rate the iteration is underdamped: it overshoots
    and rings, and the ringing decays at sqrt(momentum) per step no matter how
    the learning rate is set. At 0.9 that is 0.949 per step, which is why the
    budget below is nearly spent while plain SGD finishes in 22.
    """
    w = Tensor(START.copy())

    steps = steps_to_reach_minimum(SGD([w], lr=0.1, momentum=0.9), w, TOL)

    assert steps is not None and steps < 200
    np.testing.assert_allclose(w.data, MINIMUM, atol=TOL)


def test_adam_reaches_the_minimum():
    w = Tensor(START.copy())

    steps = steps_to_reach_minimum(Adam([w], lr=0.3), w, TOL)

    assert steps is not None and steps < 200
    np.testing.assert_allclose(w.data, MINIMUM, atol=TOL)


def test_every_optimizer_lowers_the_loss_monotonically_at_a_small_step():
    """Small enough steps and no momentum, so each update must reduce the loss.

    A sign error in an update rule can still converge from some starting points,
    so this pins the direction rather than the destination.
    """
    for make in (lambda p: SGD(p, lr=0.01), lambda p: Adam(p, lr=0.01)):
        w = Tensor(START.copy())
        optimizer = make([w])
        previous = float(quadratic(w).data)

        for _ in range(20):
            optimizer.zero_grad()
            quadratic(w).backward()
            optimizer.step()

            current = float(quadratic(w).data)
            assert current < previous
            previous = current


# SGD mechanics.


def test_plain_sgd_is_one_step_of_gradient_descent():
    w = Tensor([1.0, 2.0])
    optimizer = SGD([w], lr=0.1)

    (Tensor([3.0, 4.0]) * w).sum().backward()
    optimizer.step()

    # Gradient is [3, 4], so the step is 0.1 * [3, 4].
    np.testing.assert_allclose(w.data, [1.0 - 0.3, 2.0 - 0.4])


def test_momentum_carries_the_previous_velocity():
    """Two steps of a constant gradient, against the rule written out by hand.

    With v = momentum * v + grad, a constant gradient g gives v = g on the first
    step and v = 1.9 * g on the second. A velocity buffer that was rebound
    rather than updated in place, or reset between steps, would give g twice.
    """
    w = Tensor([0.0])
    optimizer = SGD([w], lr=0.1, momentum=0.9)

    for _ in range(2):
        optimizer.zero_grad()
        (Tensor([2.0]) * w).sum().backward()
        optimizer.step()

    # The gradient is 2 at every step, so the velocity is 2 and then
    # 0.9 * 2 + 2 = 3.8.
    np.testing.assert_allclose(w.data, [-0.1 * 2.0 - 0.1 * 3.8])


def test_momentum_keeps_moving_once_the_gradient_stops():
    """The velocity is what makes it coast, so it should coast."""
    w = Tensor([0.0])
    optimizer = SGD([w], lr=0.1, momentum=0.9)

    (Tensor([2.0]) * w).sum().backward()
    optimizer.step()
    moved = w.data.copy()

    # No new gradient this step, so anything that happens is the velocity alone.
    optimizer.zero_grad()
    optimizer.step()

    np.testing.assert_allclose(w.data, moved - 0.1 * 0.9 * 2.0)


def test_weight_decay_shrinks_a_parameter_with_no_gradient():
    """With grad 0 the update is w -= lr * weight_decay * w, a geometric decay."""
    w = Tensor([4.0, -6.0])
    optimizer = SGD([w], lr=0.1, weight_decay=0.5)

    for _ in range(10):
        optimizer.zero_grad()
        optimizer.step()

    factor = (1.0 - 0.1 * 0.5) ** 10
    np.testing.assert_allclose(w.data, np.array([4.0, -6.0]) * factor)


def test_weight_decay_moves_the_minimum_toward_zero():
    """Weight decay changes the problem being solved, so the answer changes too.

    The iteration stops where the total gradient vanishes, which is where
    ``2 * curvature * (w - minimum) + weight_decay * w = 0``. Solving gives the
    pulled-in minimum below. Worth stating as a test rather than as a comment:
    it is easy to read weight decay as a free regularizer and be surprised that
    the fixed point is no longer where the loss alone would put it.
    """
    decay = 0.5
    w = Tensor(START.copy())
    optimizer = SGD([w], lr=0.1, weight_decay=decay)

    for _ in range(500):
        optimizer.zero_grad()
        quadratic(w).backward()
        optimizer.step()

    shifted = 2.0 * CURVATURE * MINIMUM / (2.0 * CURVATURE + decay)

    np.testing.assert_allclose(w.data, shifted, atol=1e-9)
    assert np.abs(w.data).max() < np.abs(MINIMUM).max()


# Adam mechanics.


def test_adam_first_step_is_the_learning_rate_regardless_of_gradient_size():
    """What bias correction buys: a first step of lr, not of lr / 3.

    On step 1 the corrected moments are exactly grad and grad ** 2, so the ratio
    is sign(grad) and the parameter moves by lr. Without the correction the same
    step would be (1 - b1) / sqrt(1 - b2) times larger, about 3.16x, because the
    two moments are biased toward zero by different amounts.

    The smallest gradient tried is 1e-3 rather than something tinier: eps sits in
    the denominator, so once the gradient is down near eps itself the step really
    does shrink, which is what eps is there to do.
    """
    for gradient in (1e-3, 1.0, 1e6):
        w = Tensor([0.0])
        optimizer = Adam([w], lr=0.1)

        (Tensor([gradient]) * w).sum().backward()
        optimizer.step()

        np.testing.assert_allclose(w.data, [-0.1], rtol=1e-4)


def test_adam_is_invariant_to_rescaling_the_loss():
    """Multiplying the loss by 1000 multiplies every gradient by 1000.

    ``m`` scales with it and ``sqrt(v)`` scales with it too, so the ratio is
    unchanged and the trajectory is identical. This is the property that makes
    Adam insensitive to loss scaling, and it fails immediately if one moment is
    corrected and the other is not.

    Identical to a tolerance rather than exactly: eps is an absolute quantity
    sitting next to a denominator that did scale, so it is worth relatively less
    on the scaled run. That is the one part of Adam that scale invariance does
    not cover, and at these gradient magnitudes it is worth about 1e-9 a step.
    """
    plain = Tensor(START.copy())
    scaled = Tensor(START.copy())

    plain_optimizer = Adam([plain], lr=0.1)
    scaled_optimizer = Adam([scaled], lr=0.1)

    for _ in range(30):
        plain_optimizer.zero_grad()
        quadratic(plain).backward()
        plain_optimizer.step()

        scaled_optimizer.zero_grad()
        (Tensor(1000.0) * quadratic(scaled)).backward()
        scaled_optimizer.step()

    np.testing.assert_allclose(scaled.data, plain.data, rtol=1e-7)


def test_adam_bias_correction_wears_off():
    """The correction is a function of t, so it should approach 1 and stay there.

    Checked on the stored state rather than through a parameter: the buffers
    hold the uncorrected moments, and a step that wrote the corrected values
    back would compound the correction and slowly break the ratio.
    """
    w = Tensor([0.0])
    optimizer = Adam([w], lr=0.1)

    for _ in range(50):
        optimizer.zero_grad()
        (Tensor([2.0]) * w).sum().backward()
        optimizer.step()

    assert optimizer.t == 50

    # A constant gradient of 2, so the uncorrected first moment converges to 2
    # from below and the second to 4.
    np.testing.assert_allclose(optimizer.m[0], [2.0 * (1.0 - 0.9**50)])
    np.testing.assert_allclose(optimizer.v[0], [4.0 * (1.0 - 0.999**50)])


def test_adam_orbits_the_minimum_rather_than_landing_on_it():
    """Why the convergence tolerance above is 1e-4 and not machine precision.

    Near the minimum the update is close to lr * sign(grad) whatever the
    gradient's magnitude, so a constant learning rate cannot come to rest at the
    bottom: it steps across, the sign flips, and it steps back. The orbit decays
    as the running mean averages the alternating signs away, but slowly, so
    within the budget the parameter is circling the minimum at a distance of
    about 1e-4 and passing through it on the way round. That is a property of
    the algorithm rather than a defect in the implementation, and it is the
    reason real training decays the learning rate.
    """
    w = Tensor(START.copy())
    optimizer = Adam([w], lr=0.3)

    offsets = []
    for _ in range(200):
        optimizer.zero_grad()
        quadratic(w).backward()
        optimizer.step()
        offsets.append((w.data - MINIMUM).copy())

    tail = np.array(offsets[-50:])

    # Close, and getting no closer: the orbit is small but it is still an orbit.
    assert np.abs(tail).max() < 1e-2
    assert np.abs(tail[-1]).max() > 1e-9

    # It crosses the minimum instead of settling to one side of it.
    signs = np.sign(tail[:, 0])
    assert np.any(signs[1:] != signs[:-1])


# Behavior both optimizers share. Each test builds its parameters inside the
# loop, so the second optimizer starts from the same state the first one did
# rather than from wherever the first one left off.

BOTH = [
    lambda params: SGD(params, lr=0.1, momentum=0.9, weight_decay=0.01),
    lambda params: Adam(params, lr=0.1),
]


def test_zero_grad_clears_every_parameter():
    for make in BOTH:
        optimizer = make([Tensor([1.0, 2.0]), Tensor([[3.0]])])
        for param in optimizer.params:
            param.grad += 1.0

        optimizer.zero_grad()

        for param in optimizer.params:
            np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))


def test_step_reads_the_gradient_without_writing_to_it():
    """``step`` owns ``data``; ``backward`` owns ``grad``, and it stays that way.

    Weight decay is the place this could go wrong: adding the penalty into
    ``param.grad`` in place would be the same update on the first step and a
    compounding one thereafter, since the term would still be there next time.
    """
    for make in BOTH:
        optimizer = make([Tensor([1.0, 2.0])])
        param = optimizer.params[0]
        param.grad += np.array([0.5, -0.25])
        before = param.grad.copy()

        optimizer.step()
        optimizer.step()

        np.testing.assert_array_equal(param.grad, before)


def test_updates_keep_parameters_float64():
    for make in BOTH:
        optimizer = make([Tensor([1.0, 2.0])])
        param = optimizer.params[0]
        param.grad += 1.0

        optimizer.step()

        assert param.data.dtype == np.float64


def test_optimizers_handle_several_parameters_of_different_shapes():
    """Each parameter carries its own state, matched to its own shape."""
    for make in BOTH:
        optimizer = make([Tensor(np.zeros((2, 3))), Tensor(np.zeros(4))])
        for param in optimizer.params:
            param.grad += 1.0

        optimizer.step()

        assert optimizer.params[0].data.shape == (2, 3)
        assert optimizer.params[1].data.shape == (4,)
        # Every gradient was positive, so every entry moved down.
        for param in optimizer.params:
            assert np.all(param.data < 0.0)


def test_params_can_be_passed_as_any_iterable():
    """``model.parameters()`` returns a list, but a generator should work too."""
    tensors = [Tensor([1.0]), Tensor([2.0])]

    optimizer = SGD((t for t in tensors), lr=0.1)

    assert len(optimizer.params) == 2
    assert len(optimizer.velocities) == 2
