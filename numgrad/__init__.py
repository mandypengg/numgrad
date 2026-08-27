"""numgrad: reverse-mode automatic differentiation in pure NumPy.

This module is the public API surface. Tests and examples import from
``numgrad`` directly rather than reaching into submodules::

    from numgrad import Tensor, SGD

As each piece lands, re-export it here and add it to ``__all__``.
"""

from numgrad.data import batches, load_mnist, one_hot, train_test_split
from numgrad.gradcheck import check_grads
from numgrad.nn import (
    Linear,
    Module,
    ReLU,
    Sequential,
    Tanh,
    softmax_cross_entropy,
)
from numgrad.ops import (
    add,
    exp,
    getitem,
    log,
    matmul,
    max,
    mul,
    neg,
    pow,
    relu,
    reshape,
    sub,
    sum,
    tanh,
    transpose,
    truediv,
    unbroadcast,
)
from numgrad.optim import SGD, Adam
from numgrad.tensor import Tensor, no_grad
from numgrad.viz import draw_graph

__version__ = "0.0.0"

__all__: list[str] = [
    "SGD",
    "Adam",
    "Linear",
    "Module",
    "ReLU",
    "Sequential",
    "Tanh",
    "Tensor",
    "add",
    "batches",
    "check_grads",
    "draw_graph",
    "exp",
    "getitem",
    "load_mnist",
    "log",
    "matmul",
    "max",
    "mul",
    "neg",
    "no_grad",
    "one_hot",
    "pow",
    "relu",
    "reshape",
    "softmax_cross_entropy",
    "sub",
    "sum",
    "tanh",
    "train_test_split",
    "transpose",
    "truediv",
    "unbroadcast",
]
