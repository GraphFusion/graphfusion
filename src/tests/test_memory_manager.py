# src/tests/test_memory_manager.py
import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity
from graphfusion.memory_manager import MemoryManager

@pytest.fixture
def setup_memory_manager():
    mm = MemoryManager()
    embedding1 = np.random.rand(128)
    embedding2 = np.random.rand(128)
    embedding3 = np.random.rand(128)
    mm.store_in_memory(embedding1, {"patient_id": "123", "symptoms": ["fever", "cough"], "diagnosis": "flu"}, label="respiratory")
    mm.store_in_memory(embedding2, {"patient_id": "456", "symptoms": ["headache", "nausea"], "diagnosis": "migraine"}, label="neurology")
    mm.store_in_memory(embedding3, {"patient_id": "789", "symptoms": ["fatigue", "muscle pain"], "diagnosis": "viral infection"}, label="general")
    return mm

def test_store_in_memory(setup_memory_manager):
    mm = setup_memory_manager
    embedding = np.random.rand(128)
    mm.store_in_memory(embedding, {"patient_id": "000", "symptoms": ["sore throat"], "diagnosis": "cold"})
    assert len(mm.memory) == 4
    assert mm.memory[-1]["data"]["diagnosis"] == "cold"

def test_retrieve_similar(setup_memory_manager):
    mm = setup_memory_manager
    query_embedding = np.random.rand(128)
    similar_cases = mm.retrieve_similar(query_embedding, top_k=2)
    assert len(similar_cases) == 2
    assert all("embedding" in case and "data" in case for case in similar_cases)

def test_update_entry(setup_memory_manager):
    mm = setup_memory_manager
    new_data = {"patient_id": "123", "symptoms": ["fever", "cough"], "diagnosis": "common cold"}
    mm.update_entry(0, new_data)
    assert mm.memory[0]["data"]["diagnosis"] == "common cold"

def test_forget_old_entries():
    mm = MemoryManager()

    # Add embeddings with varying similarity to the reference embedding
    reference_embedding = np.random.rand(128)
    embedding1 = reference_embedding * 0.9  # Close to reference
    embedding2 = np.random.rand(128) * 0.1  # Very dissimilar
    embedding3 = np.random.rand(128) * 0.5  # Moderately dissimilar

    mm.store_in_memory(embedding1, {"data": "entry1"})
    mm.store_in_memory(embedding2, {"data": "entry2"})
    mm.store_in_memory(embedding3, {"data": "entry3"})

    # Set threshold high to remove entries not similar to reference_embedding
    mm.forget_old_entries(reference_embedding=reference_embedding, threshold=0.8)

    # Test if the memory manager correctly forgets entries
    assert len(mm.memory) == 2  # Should retain the similar embeddings


def test_update_on_feedback(setup_memory_manager):
    mm = setup_memory_manager
    feedback = {"relevance": 0.4}  # Simulate feedback indicating low relevance
    initial_embedding = mm.memory[0]["embedding"].copy()
    mm.update_on_feedback(feedback)
    updated_embedding = mm.memory[0]["embedding"]
    assert np.allclose(updated_embedding, initial_embedding * 0.9)
