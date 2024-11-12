import pytest
from graphfusion.knowledge_graph import KnowledgeGraph

def test_add_node_and_edge():
    kg = KnowledgeGraph()
    # Add node with attributes
    kg.add_node("patient_1", attributes={"name": "Patient A"})
    
    # Verify node addition
    node_data = kg.graph.nodes["patient_1"]
    assert node_data["name"] == "Patient A"

    # Add another node and an edge with a relationship type and confidence score
    kg.add_node("disease_1", attributes={"name": "Flu"})
    kg.add_edge("patient_1", "disease_1", relationship_type="diagnosed_with", confidence=0.9)
    
    # Verify edge addition and attributes
    edge_data = kg.graph.get_edge_data("patient_1", "disease_1")
    assert edge_data["type"] == "diagnosed_with"
    assert edge_data["confidence"] == 0.9

def test_update_edge():
    kg = KnowledgeGraph()
    kg.add_node("patient_1")
    kg.add_node("disease_1")
    kg.add_edge("patient_1", "disease_1", relationship_type="diagnosed_with", confidence=0.9)

    # Update the confidence score of the edge
    kg.update_edge("patient_1", "disease_1", new_confidence=0.95)

    # Verify updated confidence score
    edge_data = kg.graph.get_edge_data("patient_1", "disease_1")
    assert edge_data["confidence"] == 0.95

def test_get_related_nodes():
    kg = KnowledgeGraph()
    kg.add_node("patient_1")
    kg.add_node("disease_1")
    kg.add_edge("patient_1", "disease_1", relationship_type="diagnosed_with", confidence=0.9)

    related_nodes = kg.get_related_nodes("patient_1", relationship_type="diagnosed_with")

    # Verify the retrieved related nodes
    assert len(related_nodes) == 1
    assert related_nodes[0][0] == "disease_1"
    assert related_nodes[0][1]["type"] == "diagnosed_with"
    assert related_nodes[0][1]["confidence"] == 0.9

def test_find_path():
    kg = KnowledgeGraph()
    kg.add_node("patient_1")
    kg.add_node("disease_1")
    kg.add_node("treatment_1")
    kg.add_edge("patient_1", "disease_1", relationship_type="diagnosed_with")
    kg.add_edge("disease_1", "treatment_1", relationship_type="treated_with")

    path = kg.find_path("patient_1", "treatment_1")

    # Verify the found path
    assert path == ["patient_1", "disease_1", "treatment_1"]

def test_find_cluster():
    kg = KnowledgeGraph()
    kg.add_node("patient_1")
    kg.add_node("disease_1")
    kg.add_node("disease_2")
    kg.add_edge("patient_1", "disease_1", relationship_type="diagnosed_with")
    kg.add_edge("patient_1", "disease_2", relationship_type="diagnosed_with")

    cluster = kg.find_cluster("patient_1", radius=1)

    # Verify the cluster nodes within radius 1
    assert "patient_1" in cluster
    assert "disease_1" in cluster
    assert "disease_2" in cluster

def test_recommend_based_on_graph():
    kg = KnowledgeGraph()
    kg.add_node("patient_1")
    kg.add_node("disease_1")
    kg.add_node("disease_2")
    kg.add_edge("patient_1", "disease_1", relationship_type="diagnosed_with", confidence=0.8)
    kg.add_edge("patient_1", "disease_2", relationship_type="diagnosed_with", confidence=0.6)

    recommendations = kg.recommend_based_on_graph("patient_1", top_k=1)

    # Verify that the top recommendation is "disease_1" based on confidence score
    assert len(recommendations) == 1
    assert recommendations[0][0] == "disease_1"
    assert recommendations[0][1]["confidence"] == 0.8