"""Train a small MLP on MNIST using numgrad.

A 784 -> 128 -> 10 network with a ReLU in the middle, trained with softmax cross
entropy on the standard 60k/10k split. Every gradient in it comes from the tape
in ``numgrad.tensor``; there is nothing here that the library does not do.

Run it with::

    python examples/train_mnist.py
    python examples/train_mnist.py --optimizer adam --epochs 5

The first run downloads MNIST via ``numgrad.data`` and caches it, so later runs
start immediately. Loss and accuracy curves are written to ``assets/``, and a
second run overwrites them.

At the defaults this reaches about 97.7 percent test accuracy in ten epochs, at
a couple of seconds an epoch. Everything is float64, which is the tradeoff the
library makes on purpose: gradients a finite difference can confirm to a
relative 1e-6, and no attempt at speed.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Running a script puts its own directory on the import path, not the working
# directory, so a checkout that has not been pip-installed cannot see the
# package next door. Adding the repo root is what makes `python
# examples/train_mnist.py` work straight out of a clone.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from numgrad import (  # noqa: E402
    SGD,
    Adam,
    Linear,
    ReLU,
    Sequential,
    Tensor,
    batches,
    load_mnist,
    softmax_cross_entropy,
)

ASSETS_DIR = REPO_ROOT / "assets"

PIXELS = 784
HIDDEN = 128
CLASSES = 10

BATCH_SIZE = 64

# Momentum carries about ten times the plain gradient step at 0.9, which is why
# the SGD learning rate below is not larger.
MOMENTUM = 0.9

# One default per optimizer rather than one shared default. A learning rate of
# 0.1 is right for momentum SGD and far too large for Adam, whose update is
# normalized to roughly lr per step regardless of the gradient's size.
DEFAULT_LR = {"sgd": 0.1, "adam": 1e-3}

# Rows per forward pass at evaluation time. Nothing here needs a gradient, but
# the graph is still built, so the whole 10k test set at once would hold every
# intermediate activation alive at the same time.
EVAL_BATCH_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--optimizer",
        choices=sorted(DEFAULT_LR),
        default="sgd",
        help=f"update rule (default: sgd with momentum {MOMENTUM})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            f"learning rate (default: {DEFAULT_LR['sgd']:g} for sgd, "
            f"{DEFAULT_LR['adam']:g} for adam)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="passes over the training set (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds both the weight initialization and the batch shuffling",
    )

    return parser.parse_args()


def build_model(seed):
    """The 784 -> 128 -> 10 net, initialized from a seeded generator.

    One generator threaded through both layers rather than one per layer, so the
    entire initialization follows from --seed and a run replays exactly.
    """
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(PIXELS, HIDDEN, rng=rng),
        ReLU(),
        Linear(HIDDEN, CLASSES, rng=rng),
    )


def build_optimizer(name, params, lr):
    if name == "sgd":
        return SGD(params, lr=lr, momentum=MOMENTUM)
    return Adam(params, lr=lr)


def train_one_epoch(model, optimizer, images, labels, seed):
    """One shuffled pass over the training set. Returns the mean loss.

    The loss is averaged over examples rather than over batches: the last batch
    of an epoch is usually short, and averaging batch means would weight its
    handful of examples as heavily as a full batch's.
    """
    total_loss = 0.0
    seen = 0

    for batch_images, batch_labels in batches(images, labels, BATCH_SIZE, seed=seed):
        # Cleared first, because the backward pass accumulates and last step's
        # gradient is still sitting in every parameter.
        optimizer.zero_grad()

        logits = model(Tensor(batch_images))
        loss = softmax_cross_entropy(logits, batch_labels)
        loss.backward()

        optimizer.step()

        total_loss += float(loss.data) * batch_images.shape[0]
        seen += batch_images.shape[0]

    return total_loss / seen


def accuracy(model, images, labels):
    """Fraction of rows whose largest logit is the true class.

    Read off ``logits.data`` directly. Softmax is monotone, so the arg max of
    the probabilities is the arg max of the logits and there is no reason to
    form the probabilities at all.
    """
    correct = 0

    for batch_images, batch_labels in batches(
        images, labels, EVAL_BATCH_SIZE, shuffle=False
    ):
        logits = model(Tensor(batch_images))
        correct += int(np.sum(logits.data.argmax(axis=1) == batch_labels))

    return correct / labels.shape[0]


def save_curves(losses, accuracies, args, lr):
    """Write the loss and accuracy curves to ``assets/``."""
    # Imported here so that training does not depend on matplotlib being
    # installed, and so the backend can be set before pyplot picks one. Agg
    # writes files and never tries to open a window.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(losses) + 1)
    subtitle = f"{args.optimizer}, lr {lr:g}, batch {BATCH_SIZE}, seed {args.seed}"

    figure, axes = plt.subplots()
    axes.plot(epochs, losses, marker="o")
    axes.set_xlabel("epoch")
    axes.set_ylabel("mean training loss")
    axes.set_title(f"MNIST training loss ({subtitle})")
    axes.grid(alpha=0.3)
    # Epochs are counted, so half an epoch is not a tick a short run should get.
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    figure.savefig(ASSETS_DIR / "mnist_loss.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots()
    axes.plot(epochs, 100.0 * np.asarray(accuracies), marker="o", color="tab:green")
    axes.set_xlabel("epoch")
    axes.set_ylabel("test accuracy (%)")
    axes.set_title(f"MNIST test accuracy ({subtitle})")
    axes.grid(alpha=0.3)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure.tight_layout()
    figure.savefig(ASSETS_DIR / "mnist_accuracy.png", dpi=150)
    plt.close(figure)

    return ASSETS_DIR / "mnist_loss.png", ASSETS_DIR / "mnist_accuracy.png"


def main():
    args = parse_args()
    lr = DEFAULT_LR[args.optimizer] if args.lr is None else args.lr

    x_train, y_train, x_test, y_test = load_mnist()

    model = build_model(args.seed)
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr)

    print(
        f"{PIXELS} -> {HIDDEN} -> {CLASSES}, {args.optimizer} at lr {lr:g}, "
        f"batch {BATCH_SIZE}, {args.epochs} epochs, seed {args.seed}"
    )
    print(f"train {x_train.shape[0]}, test {x_test.shape[0]}")

    losses = []
    accuracies = []

    for epoch in range(args.epochs):
        started = time.perf_counter()

        # The seed pairs the run's seed with the epoch number, so every epoch
        # shuffles differently and the whole sequence still replays from --seed.
        loss = train_one_epoch(
            model, optimizer, x_train, y_train, seed=(args.seed, epoch)
        )
        test_accuracy = accuracy(model, x_test, y_test)

        losses.append(loss)
        accuracies.append(test_accuracy)

        print(
            f"epoch {epoch + 1:2d}/{args.epochs}  "
            f"train loss {loss:.4f}  "
            f"test accuracy {100.0 * test_accuracy:.2f}%  "
            f"({time.perf_counter() - started:.1f}s)"
        )

    for path in save_curves(losses, accuracies, args, lr):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
