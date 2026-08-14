"""Draw the computation graph of a small expression.

Writes ``assets/graph.svg``, the picture at the top of the README::

    python examples/render_graph.py

The expression is ``z = (x * y + x).relu()`` at x = 2 and y = 3, which is small
enough to read in one look and still shows the thing that is easy to get wrong.
``x`` is used twice, once by the multiply and once by the add, so two paths run
back to it and its gradient is the sum of both: 3 from ``x * y`` and 1 from
``+ x``, giving 4. That summation is the rule the whole library is built around,
and here it is a number you can check by eye.

Needs graphviz, both the Python package and the ``dot`` binary on PATH.
"""

import sys
from pathlib import Path

# Same reason as examples/train_mnist.py: a script's own directory goes on the
# import path, not the working directory, so a clone that has not been installed
# cannot see the package next door.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from numgrad import Tensor, draw_graph  # noqa: E402

ASSETS_DIR = REPO_ROOT / "assets"


def main():
    x = Tensor(2.0)
    y = Tensor(3.0)

    z = (x * y + x).relu()

    # Without this every gradient in the drawing would be 0, which is the state
    # the tape is in before anything asks it for an answer.
    z.backward()

    graph = draw_graph(z, names={"x": x, "y": y})

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / "graph.svg"

    # piped rather than rendered, so graphviz does not leave its intermediate
    # DOT file sitting in assets/ next to the picture.
    path.write_bytes(graph.pipe(format="svg"))

    print(
        f"z = {float(z.data):g}, dz/dx = {float(x.grad):g}, dz/dy = {float(y.grad):g}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
