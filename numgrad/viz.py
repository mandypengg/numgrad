"""Drawing the tape.

``backward()`` is a walk over a graph, and the fastest way to understand it is
to look at the graph. This module turns any ``Tensor`` into a Graphviz drawing
of everything it was computed from: one box per tensor showing its value and its
gradient, one ellipse per op, and an edge for every dependency the tape recorded.

Graphviz is the one thing in this project that is neither NumPy nor a test
runner, and it earns that by being a picture rather than a computation. The
import sits inside ``draw_graph`` so that ``import numgrad`` never needs it, and
nothing in the autodiff core knows this module exists. Rendering also needs the
``dot`` binary on PATH, which the Python package does not install.

The traversal below is the same iterative post-order that ``Tensor.backward()``
uses, for the same reason: a picture is drawn in the order the gradients flow.

Implemented so far: draw_graph.
"""

import numpy as np


def _sort_key(tensor):
    """Order a node's inputs deterministically.

    ``Tensor._prev`` is a set, so iterating it directly would order the inputs
    of an op by object identity, which changes between runs. Graphviz lays a
    graph out in the order it is handed, so an unsorted walk produces a
    different SVG every time and a committed drawing that churns in every diff.
    """
    return (tensor._op, tensor.data.shape, np.array2string(tensor.data))


def _sorted_prev(tensor):
    return sorted(tensor._prev, key=_sort_key)


def _walk(root):
    """Every tensor ``root`` depends on, inputs before the nodes that use them.

    An explicit stack rather than recursion, matching ``Tensor.backward()``.
    Each entry pairs a node with a flag saying whether its inputs have already
    been pushed; popping an entry with the flag set means the node is ready.
    """
    ordered = []
    visited = set()

    stack = [(root, False)]
    while stack:
        node, inputs_pushed = stack.pop()

        if inputs_pushed:
            ordered.append(node)
            continue

        # Membership by identity, so two tensors holding equal values stay two
        # nodes in the picture, exactly as they are two nodes on the tape.
        if id(node) in visited:
            continue
        visited.add(id(node))

        stack.append((node, True))
        # Reversed, so that after the stack turns the order around the inputs
        # come out in _sorted_prev order.
        for parent in reversed(_sorted_prev(node)):
            if id(parent) not in visited:
                stack.append((parent, False))

    return ordered


def _format(array):
    """A scalar as a number, anything larger as its shape.

    Printing a 784x128 weight matrix into a node label produces a drawing no one
    can read, and the shape is the part that matters at that size anyway.
    """
    if array.ndim == 0:
        return f"{float(array):.4g}"
    if array.size == 1:
        return f"{float(array.reshape(())):.4g}"
    return f"array{array.shape}"


def _tensor_label(tensor, name):
    """Name if it has one, then the value, then the gradient.

    Joined with a literal backslash-l, which is the DOT escape for a line break
    that also left-justifies the line. A real newline character would not do:
    Graphviz treats it as ordinary whitespace and runs the lines together.
    """
    lines = []
    if name is not None:
        lines.append(name)
    lines.append(f"data {_format(tensor.data)}")
    lines.append(f"grad {_format(tensor.grad)}")

    # Trailing separator too, so the last line is left-justified like the rest.
    return "".join(f"{line}\\l" for line in lines)


def draw_graph(root, names=None, rankdir="LR"):
    """Build a ``graphviz.Digraph`` of the graph behind ``root``.

    Parameters
    ----------
    root:
        The tensor to draw backwards from, usually a loss. Call ``backward()``
        first if you want the gradients in the drawing to be anything but zero.
    names:
        Optional ``{name: tensor}`` mapping, so leaves can be labelled with the
        variable they were bound to. A tensor does not know its own name, and
        ``x`` reads better than ``data 2``.
    rankdir:
        Graphviz layout direction. ``LR`` puts inputs on the left and the result
        on the right, which is the direction the forward pass runs.

    Returns
    -------
    graphviz.Digraph
        Render it with ``.pipe(format="svg")`` or ``.render()``. Both need the
        ``dot`` binary on PATH.

    Every op becomes two hops: the inputs point at an ellipse for the op, and
    the ellipse points at the tensor it produced. Drawing the op as a node of
    its own rather than as an edge label is what keeps a two-input op legible,
    since both of its inputs then meet in one visible place.
    """
    # Imported here, not at module scope. Nothing else in the library needs
    # graphviz, and importing numgrad must not require it to be installed.
    import graphviz

    labels = {} if names is None else {id(t): n for n, t in names.items()}

    graph = graphviz.Digraph(
        "numgrad",
        graph_attr={"rankdir": rankdir, "bgcolor": "transparent"},
        node_attr={"fontname": "Helvetica", "fontsize": "11"},
        edge_attr={"color": "#555555", "arrowsize": "0.7"},
    )

    nodes = _walk(root)

    # Node names are positions in the traversal rather than ids, so the same
    # graph produces the same DOT source on every run.
    ids = {id(node): str(position) for position, node in enumerate(nodes)}

    for node in nodes:
        graph.node(
            ids[id(node)],
            label=_tensor_label(node, labels.get(id(node))),
            shape="box",
            style="filled,rounded",
            fillcolor="#eef3fa",
            color="#5b7fae",
        )

        # A leaf has no op and no inputs, so there is nothing more to draw.
        if not node._op:
            continue

        op_name = f"{ids[id(node)]}_{node._op}"
        graph.node(
            op_name,
            label=node._op,
            shape="ellipse",
            style="filled",
            fillcolor="#f6e6cc",
            color="#b98b3d",
        )
        graph.edge(op_name, ids[id(node)])

        for parent in _sorted_prev(node):
            graph.edge(ids[id(parent)], op_name)

    return graph
