import pytest
from unittest.mock import MagicMock
import numpy as np
from graphfusion.nmn_core import NeuralMemoryNetwork, MemoryManager, RecommendationEngine, KnowledgeGraph

# Test 1: Ensure NeuralMemoryNetwork initializes correctly
def test_nmn_initialization():
    # Test initialization with default parameters (i.e., memory_manager, recommendation_engine, knowledge_graph should be created)
    nmn = NeuralMemoryNetwork()

    # Ensure that components are initialized correctly
    assert isinstance(nmn.memory_manager, MemoryManager)
    assert isinstance(nmn.recommendation_engine, RecommendationEngine)
    assert isinstance(nmn.knowledge_graph, KnowledgeGraph)

# Test 2: Ensure the embedding generation works
def test_generate_embedding():
    nmn = NeuralMemoryNetwork()

    # Test input text
    text = "This is a test sentence."
    
    embedding = nmn.generate_embedding(text)
    
    # Ensure the embedding is a numpy array
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 768)  # Output of distilbert should be of shape (1, 768)

# Test 3: Store data in memory and update knowledge graph
def test_store_in_memory():
    nmn = NeuralMemoryNetwork()

    # Test input data
    text = "Test case for memory storage."
    data = {"case_id": 1, "description": "Test case"}
    
    # Mock the knowledge graph's add_node and add_edge methods to avoid actual graph manipulation
    nmn.knowledge_graph.add_node = MagicMock(return_value="node_123")
    nmn.knowledge_graph.add_edge = MagicMock()

    # Store in memory
    nmn.store_in_memory(text, data)
    
    # Ensure add_node and add_edge are called
    nmn.knowledge_graph.add_node.assert_called_once_with(data, label=None)
    assert nmn.knowledge_graph.add_edge.call_count > 0

# Test 4: Retrieve similar cases from memory and knowledge graph
def test_retrieve_similar():
    nmn = NeuralMemoryNetwork()

    # Mocking memory_manager and knowledge_graph methods
    nmn.memory_manager.retrieve_similar = MagicMock(return_value=[{"case_id": 1, "description": "Test case"}])
    nmn.knowledge_graph.retrieve_linked_nodes = MagicMock(return_value=[{"case_id": 2, "description": "Linked case"}])

    text = "Test case retrieval."
    similar_cases = nmn.retrieve_similar(text)
    
    # Verify similar cases are returned correctly
    assert len(similar_cases) > 0
    assert similar_cases[0]["case_id"] == 1  # Ensure the mock data is retrieved

# Test 5: Generate recommendations based on similar cases
def test_generate_recommendations():
    nmn = NeuralMemoryNetwork()

    # Mocking the necessary methods
    nmn.retrieve_similar = MagicMock(return_value=[{"case_id": 1, "description": "Test case"}])
    nmn.recommendation_engine.generate_recommendations = MagicMock(return_value=["Recommended treatment A", "Recommended treatment B"])

    text = "Generate recommendations for treatment."
    recommendations = nmn.generate_recommendations(text)

    # Ensure that recommendations are returned correctly
    assert len(recommendations) == 2
    assert recommendations == ["Recommended treatment A", "Recommended treatment B"]

# Test 6: Update memory and knowledge graph on feedback
def test_update_on_feedback():
    nmn = NeuralMemoryNetwork()

    # Mocking the update methods
    nmn.memory_manager.update_on_feedback = MagicMock()
    nmn.knowledge_graph.reduce_edge_weight = MagicMock()

    feedback = [{"node_id": "node_123", "relevance": 0.4}]
    nmn.update_on_feedback(feedback)
    
    # Ensure that the memory manager and knowledge graph methods are called
    nmn.memory_manager.update_on_feedback.assert_called_once_with(feedback)
    nmn.knowledge_graph.reduce_edge_weight.assert_called_once_with("node_123")

