import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import networkx as nx
from graphfusionai_project.utils import Utils

def test_validate_graph():
    g = nx.DiGraph()
    g.add_edge("A", "B")
    try:
        Utils.validate_graph(g)
    except Exception:
        pytest.fail("Graph validation failed.")

def test_export_graph_json(tmp_path):
    g = nx.DiGraph()
    g.add_edge("A", "B", relation="test")
    file_path = tmp_path / "graph.json"
    Utils.export_graph(g, str(file_path), format="json")
    assert file_path.exists()
