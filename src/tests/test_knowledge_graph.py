import pytest
from graphfusion.knowledge_graph import KnowledgeGraph

def test_add_node_and_edge():
    kg = KnowledgeGraph()
    kg.add_node("patient_1", data={"name": "Patient A"})
    kg.add_node("diagnosis", data={"type": "condition"})
    kg.add_edge("patient_1", "diagnosis", relation="has")
    assert "diagnosis" in kg.find_related("patient_1")
