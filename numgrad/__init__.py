"""numgrad: reverse-mode automatic differentiation in pure NumPy.

This module is the public API surface. Tests and examples import from
``numgrad`` directly rather than reaching into submodules::

    from numgrad import Tensor, SGD

As each piece lands, re-export it here and add it to ``__all__``.
"""

from numgrad.gradcheck import check_grads
from numgrad.nn import softmax_cross_entropy
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
from numgrad.tensor import Tensor

__version__ = "0.0.0"

__all__: list[str] = [
    "Tensor",
    "add",
    "check_grads",
    "exp",
    "getitem",
    "log",
    "matmul",
    "max",
    "mul",
    "neg",
    "pow",
    "relu",
    "reshape",
    "softmax_cross_entropy",
    "sub",
    "sum",
    "tanh",
    "transpose",
    "truediv",
    "unbroadcast",
]
