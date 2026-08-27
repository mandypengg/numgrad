"""Tests for the pieces in ``numgrad.nn``.

The losses here are composed from ops rather than carrying gradients of their
own, so the gradient checks are checking the composition: that the graph a loss
builds differentiates to the right thing, and that it stays numerically sound at
logit magnitudes where the textbook formula does not.

The same goes for the layers: a ``Linear`` has no backward pass of its own, so
what the gradient checks here confirm is that stacking layers composes the ops'
gradients correctly, in particular that the bias broadcast across the batch is
undone on the way back.
"""

import numpy as np
import pytest

from numgrad import (
    Linear,
    Module,
    ReLU,
    Sequential,
    Tanh,
    Tensor,
    check_grads,
    no_grad,
    softmax_cross_entropy,
)


def reference_loss(logits, labels):
    """Mean cross entropy computed directly in NumPy, for comparison.

    Written the stable way, since the point of comparing against it is the value
    of the loss and not the arithmetic used to reach it.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return -np.mean(np.log(probs[np.arange(logits.shape[0]), labels]))


def reference_loss_in_log_space(logits, labels):
    """The same loss via ``np.logaddexp``, which never forms a probability.

    ``reference_loss`` divides and then logs, so a true class whose probability
    underflows to exactly 0 sends it to inf. Accumulating in log space instead
    keeps that case finite, and it is an implementation NumPy provides rather
    than a restatement of the one under test.
    """
    rows = np.arange(logits.shape[0])
    return np.mean(np.logaddexp.reduce(logits, axis=1) - logits[rows, labels])


def reference_grad(logits, labels):
    """d(loss)/d(logits), worked out by hand: (softmax - one hot) / batch."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(logits.shape[0]), labels] = 1.0

    return (probs - one_hot) / logits.shape[0]


LABELS = np.array([0, 3, 1, 2, 2, 0])


def test_forward_matches_the_reference():
    rng = np.random.default_rng(40)
    logits = Tensor(rng.standard_normal((6, 4)))

    loss = softmax_cross_entropy(logits, LABELS)

    assert loss.shape == ()
    np.testing.assert_allclose(loss.data, reference_loss(logits.data, LABELS))


def test_uniform_logits_give_log_of_the_class_count():
    """Every class equally likely, so the loss is log(classes) exactly."""
    logits = Tensor(np.zeros((3, 5)))

    loss = softmax_cross_entropy(logits, np.array([0, 2, 4]))

    np.testing.assert_allclose(loss.data, np.log(5.0))


def test_a_confident_correct_prediction_costs_almost_nothing():
    logits = Tensor([[50.0, 0.0, 0.0], [0.0, 0.0, 50.0]])

    loss = softmax_cross_entropy(logits, np.array([0, 2]))

    assert 0.0 <= float(loss.data) < 1e-15


def test_gradient_matches_the_hand_derived_form():
    """The analytic gradient is (softmax - one hot) / batch.

    Worth pinning against the closed form as well as against a finite
    difference: the two disagree in different ways, and a composition that was
    off by the batch factor would still pass a check that only compared shapes.
    """
    rng = np.random.default_rng(41)
    logits = Tensor(rng.standard_normal((6, 4)))

    softmax_cross_entropy(logits, LABELS).backward()

    np.testing.assert_allclose(logits.grad, reference_grad(logits.data, LABELS))


def test_the_rows_of_the_gradient_sum_to_zero():
    """Adding a constant to a row cannot change the loss, so each row sums to 0.

    This is the shift invariance the log-sum-exp trick relies on, read off the
    gradient. It also means the row max carries no gradient of its own, which is
    why it can be left on the tape.
    """
    rng = np.random.default_rng(42)
    logits = Tensor(rng.standard_normal((6, 4)))

    softmax_cross_entropy(logits, LABELS).backward()

    np.testing.assert_allclose(logits.grad.sum(axis=1), np.zeros(6), atol=1e-15)


def test_shifting_a_row_leaves_the_loss_unchanged():
    rng = np.random.default_rng(43)
    logits = rng.standard_normal((6, 4))
    shifts = rng.standard_normal((6, 1))

    plain = softmax_cross_entropy(Tensor(logits), LABELS)
    shifted = softmax_cross_entropy(Tensor(logits + shifts), LABELS)

    np.testing.assert_allclose(plain.data, shifted.data)


def test_gradcheck_softmax_cross_entropy():
    rng = np.random.default_rng(44)
    logits = Tensor(rng.standard_normal((6, 4)))

    check_grads(lambda x: softmax_cross_entropy(x, LABELS), [logits])


def test_gradcheck_with_every_row_sharing_a_label():
    """One class for the whole batch, so its column takes every one-hot term."""
    rng = np.random.default_rng(45)
    logits = Tensor(rng.standard_normal((6, 4)))

    check_grads(lambda x: softmax_cross_entropy(x, np.zeros(6, dtype=int)), [logits])


def test_gradcheck_on_a_single_row_batch():
    rng = np.random.default_rng(46)
    logits = Tensor(rng.standard_normal((1, 5)))

    check_grads(lambda x: softmax_cross_entropy(x, np.array([3])), [logits])


# The reason the loss subtracts the row max at all. Logits this large are not
# exotic: an untrained net with a bad initialization produces them in the first
# few steps, and the naive formula returns nan there rather than a large loss.


def test_huge_logits_stay_finite_and_differentiable():
    """Logits of magnitude 1e3, where softmax-then-log overflows.

    ``exp(1e3)`` is ``inf`` in float64, so forming the softmax first gives
    ``inf / inf``, and the nan that produces flows through the backward pass and
    destroys every gradient in the graph. Subtracting the row max caps the
    largest exponent at ``exp(0)``, so nothing here is ever exponentiated above
    1 and the loss comes out at the same size as it would for small logits.
    """
    rng = np.random.default_rng(47)
    logits = Tensor(1e3 + rng.standard_normal((6, 4)))

    # The naive path really does overflow at these inputs, which is what makes
    # this test worth having rather than a restatement of the small-logit one.
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.exp(logits.data)).any()

    loss = softmax_cross_entropy(logits, LABELS)

    assert np.isfinite(loss.data)
    np.testing.assert_allclose(loss.data, reference_loss(logits.data, LABELS))

    # And the gradient is not merely finite, it is still correct.
    check_grads(lambda x: softmax_cross_entropy(x, LABELS), [logits])


def test_huge_logits_that_are_also_far_apart():
    """Magnitude 1e3 with a spread to match, so the probabilities underflow.

    An underflowed exp is a probability that genuinely was negligible, so the
    loss and the gradient are still right. Note which reference this compares
    against: the loss here is around 940, meaning the true class had probability
    exp(-940), which is 0 in float64. Forming that probability and then taking
    its log gives inf, so ``reference_loss`` cannot describe this case, while
    the loss under test never forms it and stays finite.

    There is no ``check_grads`` call here: the loss is of order 1e3 while the
    gradients being measured are of order 1e-8, and differencing the one to
    recover the other costs more digits than float64 has. That is a limit of
    finite differencing, not of the backward pass, so the gradient is compared
    against the closed form instead.
    """
    rng = np.random.default_rng(48)
    logits = Tensor(1e3 * rng.standard_normal((6, 4)))

    loss = softmax_cross_entropy(logits, LABELS)
    loss.backward()

    assert np.isfinite(loss.data)
    np.testing.assert_allclose(
        loss.data, reference_loss_in_log_space(logits.data, LABELS)
    )

    # The textbook formula gives up here, which is the whole point.
    with np.errstate(divide="ignore"):
        assert not np.isfinite(reference_loss(logits.data, LABELS))

    assert np.isfinite(logits.grad).all()
    # Compared with an absolute tolerance, because the entries that disagree
    # with the reference at all disagree at magnitudes around 1e-191: a class
    # whose probability underflowed contributes a gradient far below anything
    # that could matter, and the two routes to it round differently. A relative
    # comparison would call 0 against 1e-191 an infinite error.
    np.testing.assert_allclose(
        logits.grad, reference_grad(logits.data, LABELS), atol=1e-12
    )


def test_very_negative_logits_stay_finite():
    """The other end of the range, where exp underflows to 0 rather than to inf.

    The true class still has to keep its log-probability finite: the shifted max
    is exactly 0, so the row sum is at least 1 and its log cannot diverge.
    """
    logits = Tensor(-1e3 + np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]))

    loss = softmax_cross_entropy(logits, np.array([0, 2]))

    assert np.isfinite(loss.data)
    np.testing.assert_allclose(loss.data, reference_loss(logits.data, np.array([0, 2])))


def test_loss_and_gradients_are_float64():
    logits = Tensor([[1, 2, 3], [4, 5, 6]])

    loss = softmax_cross_entropy(logits, np.array([0, 2]))
    loss.backward()

    assert loss.data.dtype == np.float64
    assert logits.grad.dtype == np.float64


def test_the_input_logits_are_not_mutated():
    """The row max is subtracted into a new tensor, not in place."""
    rng = np.random.default_rng(49)
    logits = Tensor(rng.standard_normal((6, 4)))
    before = logits.data.copy()

    softmax_cross_entropy(logits, LABELS).backward()

    np.testing.assert_array_equal(logits.data, before)


def test_wrong_shapes_are_rejected():
    logits = Tensor(np.zeros((6, 4)))

    with pytest.raises(ValueError, match="2-D logits"):
        softmax_cross_entropy(Tensor(np.zeros(4)), np.array([0]))

    with pytest.raises(ValueError, match="one label per row"):
        softmax_cross_entropy(logits, np.array([0, 1]))


# Layers. A layer is a container for parameter tensors plus a forward pass built
# from ops, so these tests cover three things: the shapes and values a layer
# produces, the initialization it starts from, and whether gradients survive the
# trip back through a stack of them.


def test_linear_forward_is_the_affine_map():
    rng = np.random.default_rng(60)
    layer = Linear(4, 3, rng=rng)
    x = Tensor(rng.standard_normal((6, 4)))

    out = layer(x)

    assert out.shape == (6, 3)
    np.testing.assert_allclose(out.data, x.data @ layer.weight.data + layer.bias.data)


def test_linear_bias_starts_at_zeros():
    layer = Linear(4, 3, rng=np.random.default_rng(61))

    np.testing.assert_array_equal(layer.bias.data, np.zeros(3))


def test_he_initialization_has_the_right_scale():
    """Weight entries have standard deviation sqrt(2 / fan_in).

    Checked over a large fan_in so the sample standard deviation is close to the
    population one. The tolerance is loose on purpose: this is asserting that
    the scale is 2 / fan_in rather than 1 / fan_in or 2 / fan_out, all of which
    would look identical on a small square layer.
    """
    fan_in = 4096
    layer = Linear(fan_in, 64, rng=np.random.default_rng(62))

    expected = np.sqrt(2.0 / fan_in)

    assert layer.weight.shape == (fan_in, 64)
    np.testing.assert_allclose(layer.weight.data.std(), expected, rtol=0.02)
    np.testing.assert_allclose(layer.weight.data.mean(), 0.0, atol=0.01 * expected)


def test_he_initialization_holds_activation_scale_across_layers():
    """The property the scale exists for: a deep ReLU stack neither dies nor blows up.

    Twenty layers is enough for the wrong constant to show. At 1 / fan_in the
    activations would shrink by about sqrt(2) per layer, which over twenty
    layers is a factor of a thousand.
    """
    rng = np.random.default_rng(63)
    layers = []
    for _ in range(20):
        layers.append(Linear(128, 128, rng=rng))
        layers.append(ReLU())
    net = Sequential(*layers)

    x = Tensor(rng.standard_normal((64, 128)))

    out = net(x)

    # Compared against the input's own scale, since the claim is preservation
    # and not a particular absolute value.
    ratio = out.data.std() / x.data.std()
    assert 0.5 < ratio < 2.0


def test_linear_parameters_are_the_weight_and_bias():
    layer = Linear(4, 3, rng=np.random.default_rng(64))

    params = layer.parameters()

    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias


def test_activations_have_no_parameters():
    assert ReLU().parameters() == []
    assert Tanh().parameters() == []


def test_relu_and_tanh_forward():
    x = Tensor([[-2.0, 0.0, 3.0]])

    np.testing.assert_allclose(ReLU()(x).data, [[0.0, 0.0, 3.0]])
    np.testing.assert_allclose(Tanh()(x).data, np.tanh([[-2.0, 0.0, 3.0]]))


def test_sequential_applies_layers_in_order():
    rng = np.random.default_rng(65)
    first = Linear(4, 5, rng=rng)
    second = Linear(5, 3, rng=rng)
    net = Sequential(first, Tanh(), second)

    x = Tensor(rng.standard_normal((6, 4)))

    expected = np.tanh(x.data @ first.weight.data + first.bias.data)
    expected = expected @ second.weight.data + second.bias.data

    np.testing.assert_allclose(net(x).data, expected)


def test_sequential_collects_parameters_in_layer_order():
    rng = np.random.default_rng(66)
    first = Linear(4, 5, rng=rng)
    second = Linear(5, 3, rng=rng)
    net = Sequential(first, Tanh(), second, Tanh())

    params = net.parameters()

    expected = [first.weight, first.bias, second.weight, second.bias]

    assert len(params) == len(expected)
    for got, want in zip(params, expected):
        assert got is want


def test_zero_grad_clears_every_parameter():
    """And clears in place, so a held reference to .grad sees the reset."""
    rng = np.random.default_rng(67)
    net = Sequential(Linear(4, 5, rng=rng), Tanh(), Linear(5, 3, rng=rng))
    x = Tensor(rng.standard_normal((6, 4)))

    net(x).sum().backward()

    assert any(np.any(p.grad != 0.0) for p in net.parameters())

    held = net.parameters()[0].grad
    net.zero_grad()

    for param in net.parameters():
        np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))
    assert held is net.parameters()[0].grad


def test_module_without_a_forward_says_so():
    class Empty(Module):
        pass

    with pytest.raises(NotImplementedError, match="Empty"):
        Empty()(Tensor([1.0]))


def test_parameters_are_float64():
    layer = Linear(4, 3, rng=np.random.default_rng(68))

    assert layer.weight.data.dtype == np.float64
    assert layer.bias.data.dtype == np.float64


# Labels for the 4 -> 5 -> 3 net below. Three classes, so every label is < 3.
NET_LABELS = np.array([0, 2, 1, 2, 0, 1])


def build_net(seed):
    """A 4 -> 5 -> 3 net with tanh activations, and a batch of 6 inputs."""
    rng = np.random.default_rng(seed)
    net = Sequential(Linear(4, 5, rng=rng), Tanh(), Linear(5, 3, rng=rng), Tanh())
    x = Tensor(rng.standard_normal((6, 4)))
    return net, x


def test_gradcheck_a_tanh_net_end_to_end():
    """Every weight and bias in the stack, against a finite difference.

    ``check_grads`` perturbs the tensors it is handed, and those are the same
    objects the layers hold, so the forward pass inside the closure sees each
    perturbation. The closure takes the parameters positionally and ignores
    them for that reason.
    """
    net, x = build_net(70)

    check_grads(lambda *params: net(x).sum(), net.parameters())


def test_gradcheck_a_tanh_net_through_the_loss():
    """The same net with softmax cross entropy on top, which is the real graph."""
    net, x = build_net(71)

    check_grads(
        lambda *params: softmax_cross_entropy(net(x), NET_LABELS), net.parameters()
    )


def test_gradcheck_flows_back_to_the_input():
    """Gradients reach the input too, which is what makes stacking work at all."""
    net, x = build_net(72)

    check_grads(lambda inp: softmax_cross_entropy(net(inp), NET_LABELS), [x])


def test_gradcheck_a_relu_net():
    """Same shape, ReLU instead of tanh.

    ReLU takes subgradient 0 at exactly 0 and a central difference straddling 0
    disagrees with any choice made there, so the test asserts first that no
    pre-activation sits within the step size of the kink. Without that check a
    seed change could turn this into an intermittent failure that looks like a
    bug in the backward pass.
    """
    rng = np.random.default_rng(73)
    first = Linear(4, 5, rng=rng)
    second = Linear(5, 3, rng=rng)
    net = Sequential(first, ReLU(), second, ReLU())
    x = Tensor(rng.standard_normal((6, 4)))

    hidden = x.data @ first.weight.data + first.bias.data
    output = np.maximum(hidden, 0.0) @ second.weight.data + second.bias.data

    assert np.abs(hidden).min() > 1e-3
    assert np.abs(output).min() > 1e-3

    check_grads(
        lambda *params: softmax_cross_entropy(net(x), NET_LABELS), net.parameters()
    )


def test_the_bias_gradient_sums_over_the_batch():
    """One bias entry is used once per row, so its gradient is the column sum.

    This is the broadcast being undone. A backward pass that returned the
    gradient un-summed would have the wrong shape and fail loudly; one that
    averaged instead of summing would have the right shape and be quietly wrong
    by a factor of the batch size.
    """
    rng = np.random.default_rng(74)
    layer = Linear(4, 3, rng=rng)
    x = Tensor(rng.standard_normal((6, 4)))
    weights = rng.standard_normal((6, 3))

    (layer(x) * Tensor(weights)).sum().backward()

    assert layer.bias.grad.shape == (3,)
    np.testing.assert_allclose(layer.bias.grad, weights.sum(axis=0))


def test_a_parameter_used_twice_accumulates_both_contributions():
    """The same layer applied twice in one graph gets one contribution per use."""
    rng = np.random.default_rng(75)
    layer = Linear(3, 3, rng=rng)
    x = Tensor(rng.standard_normal((5, 3)))

    once = layer(x)
    once.sum().backward()
    single = layer.weight.grad.copy()

    layer.zero_grad()
    (layer(x) + layer(x)).sum().backward()

    np.testing.assert_allclose(layer.weight.grad, 2.0 * single)


def test_the_forward_pass_does_not_mutate_its_input():
    net, x = build_net(76)
    before = x.data.copy()

    softmax_cross_entropy(net(x), NET_LABELS).backward()

    np.testing.assert_array_equal(x.data, before)


# no_grad and detach through a whole model, which is where they get used.


def test_no_grad_gives_a_model_the_same_predictions():
    """Evaluation is a forward pass, and no_grad changes nothing about it."""
    net, x = build_net(77)

    recorded = net(x)
    with no_grad():
        skipped = net(x)

    np.testing.assert_array_equal(recorded.data, skipped.data)


def test_no_grad_leaves_every_parameter_gradient_at_zero():
    net, x = build_net(78)

    with no_grad():
        loss = softmax_cross_entropy(net(x), NET_LABELS)

    loss.backward()

    for param in net.parameters():
        np.testing.assert_array_equal(param.grad, np.zeros_like(param.data))


def test_no_grad_builds_no_tape_for_a_whole_net():
    """What the block actually saves: the intermediate nodes are never linked."""
    net, x = build_net(79)

    with no_grad():
        out = net(x)

    assert out._prev == set()


def test_a_detached_input_gets_no_gradient_but_the_weights_still_do():
    """The usual reason to detach: stop at the input, keep training the layer."""
    net, x = build_net(80)

    softmax_cross_entropy(net(x.detach()), NET_LABELS).backward()

    np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))
    for param in net.parameters():
        assert np.any(param.grad != 0.0)
