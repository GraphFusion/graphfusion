import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import pytest
import numpy as np
from unittest.mock import MagicMock
from graphfusion.nmn_core import NeuralMemoryNetwork
from graphfusion.memory_manager import MemoryManager
from graphfusion.recommendation_engine import RecommendationEngine
from graphfusion.knowledge_graph import KnowledgeGraph

@pytest.fixture
def setup_nmn():
    memory_manager = MemoryManager()
    recommendation_engine = RecommendationEngine()
    knowledge_graph = KnowledgeGraph()

    # Create an instance of the NeuralMemoryNetwork
    nmn = NeuralMemoryNetwork(
        memory_manager=memory_manager,
        recommendation_engine=recommendation_engine,
        knowledge_graph=knowledge_graph
    )

    return nmn, memory_manager, recommendation_engine, knowledge_graph

def test_generate_embedding(setup_nmn):
    nmn, _, _, _ = setup_nmn
    
    # Test embedding generation from text
    text = "This is a test."
    embedding = nmn.generate_embedding(text)

    assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array."
    assert embedding.shape[0] > 0, "Embedding should have a non-zero length."

def test_store_in_memory(setup_nmn):
    nmn, memory_manager, _, knowledge_graph = setup_nmn
    
    # Mocking the knowledge graph to return a fake node ID
    knowledge_graph.add_node = MagicMock(return_value="node_1")
    
    # Store data in memory and update the knowledge graph
    text = "Patient presents with a fever."
    data = {"symptom": "fever", "severity": "high"}
    nmn.store_in_memory(text, data, label="fever")

    # Check if the memory manager received the store request
    assert len(memory_manager.memory) == 1, "Memory should contain one item."
    assert memory_manager.memory[0]["data"] == data, "Stored data should match the input data."

    # Check if the knowledge graph was updated
    knowledge_graph.add_node.assert_called_once_with(data, label="fever")
    assert knowledge_graph.add_node.call_count == 1, "Knowledge graph should have one node added."

def test_retrieve_similar(setup_nmn):
    nmn, memory_manager, _, _ = setup_nmn
    
    # Store some example data in memory
    text1 = "Patient has a fever."
    data1 = {"symptom": "fever", "severity": "moderate"}
    nmn.store_in_memory(text1, data1)

    text2 = "Patient experiences chills."
    data2 = {"symptom": "chills", "severity": "moderate"}
    nmn.store_in_memory(text2, data2)

    # Retrieve similar cases
    query_text = "Patient reports a fever."
    similar_cases = nmn.retrieve_similar(query_text, top_k=2)

    assert len(similar_cases) > 0, "There should be at least one similar case retrieved."
    assert similar_cases[0]["data"] == data1, "The most similar case should match the stored data."

def test_generate_recommendations(setup_nmn):
    nmn, _, recommendation_engine, _ = setup_nmn
    
    # Mock the recommendation engine
    recommendation_engine.generate_recommendations = MagicMock(return_value=["Recommendation 1", "Recommendation 2"])

    # Generate recommendations based on text
    text = "Patient has a headache and fever."
    recommendations = nmn.generate_recommendations(text, top_k=2)

    # Ensure the recommendations are generated correctly
    assert len(recommendations) == 2, "There should be two recommendations."
    assert recommendations == ["Recommendation 1", "Recommendation 2"], "Recommendations should match the mocked return values."

def test_update_on_feedback(setup_nmn):
    nmn, memory_manager, _, knowledge_graph = setup_nmn
    
    # Mock the methods for memory and knowledge graph updates
    memory_manager.update_on_feedback = MagicMock()
    knowledge_graph.reduce_edge_weight = MagicMock()

    feedback = [
        {"node_id": "node_1", "relevance": 0.4},
        {"node_id": "node_2", "relevance": 0.8}
    ]
    
    # Update the network based on feedback
    nmn.update_on_feedback(feedback)
    
    # Ensure the memory update method was called
    memory_manager.update_on_feedback.assert_called_once_with(feedback)

    # Ensure the knowledge graph adjustment was made
    knowledge_graph.reduce_edge_weight.assert_called_once_with("node_1")

