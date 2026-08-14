"""numgrad: reverse-mode automatic differentiation in pure NumPy.

This module is the public API surface. Tests and examples import from
``numgrad`` directly rather than reaching into submodules::

    from numgrad import Tensor, SGD

As each piece lands, re-export it here and add it to ``__all__``.
"""

from numgrad.ops import add, matmul, mul, transpose
from numgrad.tensor import Tensor

__version__ = "0.0.0"

__all__: list[str] = [
    "Tensor",
    "add",
    "matmul",
    "mul",
    "transpose",
]
