"""Tests for the drawing in ``numgrad.viz``.

Almost everything here is asserted against the DOT source rather than against a
rendered image. DOT is text, it is what the module actually produces, and
checking it needs only the Python package; the ``dot`` binary that turns it into
a picture is a separate install and only one test needs it.

The structural claim worth testing is the one the drawing exists to show. In
``z = (x * y + x).relu()`` the tensor ``x`` is used twice, and it has to appear
as one node with two edges leaving it. A drawing that gave each use its own node
would look tidier and would misrepresent the tape, which is precisely the thing
a reader is looking at the picture to understand.
"""

import shutil

import numpy as np
import pytest

from numgrad import Tensor, draw_graph

# The whole module needs the Python package. The binary is checked separately.
pytest.importorskip("graphviz")


def expression():
    """``z = (x * y + x).relu()`` at x = 2, y = 3, with gradients filled in."""
    x = Tensor(2.0)
    y = Tensor(3.0)

    z = (x * y + x).relu()
    z.backward()

    return x, y, z


def test_the_ops_appear_as_nodes():
    _, _, z = expression()

    source = draw_graph(z).source

    for op in ("mul", "add", "relu"):
        assert f"label={op}" in source


def test_every_tensor_on_the_tape_gets_a_node():
    """Five tensors: x, y, x * y, x * y + x, and the relu of that."""
    _, _, z = expression()

    source = draw_graph(z).source

    # Tensor nodes are the boxes; op nodes are the ellipses.
    assert source.count("shape=box") == 5
    assert source.count("shape=ellipse") == 3


def test_a_tensor_used_twice_is_one_node_with_two_edges():
    """The point of the picture: x feeds both the multiply and the add."""
    x, _, z = expression()

    graph = draw_graph(z, names={"x": x})
    lines = graph.source.splitlines()

    # x is node 0, the first node in the traversal, so its outgoing edges are
    # the lines starting `0 ->`.
    leaving_x = [line for line in lines if line.strip().startswith("0 ->")]

    assert len(leaving_x) == 2
    assert any("mul" in line for line in leaving_x)
    assert any("add" in line for line in leaving_x)


def test_values_and_gradients_are_in_the_labels():
    """x = 2 with gradient 4, which is 3 from the mul path plus 1 from the add."""
    x, y, z = expression()

    source = draw_graph(z, names={"x": x, "y": y}).source

    assert float(x.grad) == 4.0
    assert "x\\ldata 2\\lgrad 4\\l" in source
    assert "y\\ldata 3\\lgrad 2\\l" in source


def test_an_untouched_graph_draws_zero_gradients():
    """Before backward() every gradient really is 0, and the drawing says so."""
    x = Tensor(2.0)
    z = (x * Tensor(3.0)).relu()

    source = draw_graph(z, names={"x": x}).source

    assert "x\\ldata 2\\lgrad 0\\l" in source


def test_names_are_optional():
    _, _, z = expression()

    source = draw_graph(z).source

    assert "data 2\\lgrad 4\\l" in source


def test_large_arrays_show_their_shape_rather_than_their_entries():
    """A 784x128 weight matrix in a node label would be unreadable."""
    weight = Tensor(np.zeros((784, 128)))
    x = Tensor(np.zeros((8, 784)))

    source = draw_graph((x @ weight).sum(), names={"weight": weight}).source

    assert "array(784, 128)" in source
    assert "0. 0. 0." not in source


def test_the_source_is_stable_across_runs():
    """``_prev`` is a set, so an unsorted walk would reorder between runs.

    Built twice from scratch rather than drawn twice from one graph: identical
    tensors get different ids on the second build, and it is exactly that
    difference a set-ordered traversal would leak into the output.
    """
    first = draw_graph(expression()[2]).source
    second = draw_graph(expression()[2]).source

    assert first == second


def test_rankdir_is_configurable():
    _, _, z = expression()

    assert "rankdir=LR" in draw_graph(z).source
    assert "rankdir=TB" in draw_graph(z, rankdir="TB").source


@pytest.mark.skipif(shutil.which("dot") is None, reason="needs the dot binary")
def test_it_renders_to_svg():
    """The one test that shells out to Graphviz itself."""
    _, _, z = expression()

    svg = draw_graph(z).pipe(format="svg").decode()

    assert svg.startswith("<?xml")
    assert "<svg" in svg
    assert "relu" in svg
