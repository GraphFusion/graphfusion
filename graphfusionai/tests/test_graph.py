import pytest
from graphfusionai.graph import Graph

def test_add_knowledge():
    g = Graph()
    g.add_knowledge("A", "connected_to", "B")
    assert ("A", "B") in g.graph.edges
    assert g.graph["A"]["B"]["relation"] == "connected_to"

def test_query_graph():
    g = Graph()
    g.add_knowledge("A", "parent_of", "B")
    g.add_knowledge("B", "sibling_of", "C")
    results = g.query_graph(source="A")
    assert ("A", "parent_of", "B") in results
    assert len(results) == 1

def test_visualize_graph():
    g = Graph()
    g.add_knowledge("A", "connected_to", "B")
    try:
        g.visualize()
    except Exception:
        pytest.fail("Graph visualization failed.")
