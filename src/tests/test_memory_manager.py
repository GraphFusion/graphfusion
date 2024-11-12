import pytest
from graphfusion.memory_manager import MemoryManager
from graphfusion.nmn_core import NeuralMemoryNetwork

def test_add_and_retrieve():
    memory = MemoryManager()
    nmn = NeuralMemoryNetwork()
    embedding = nmn.embed_text("medical case data")
    memory.add_to_memory({"embedding": embedding, "text": "medical case data"})
    retrieved = memory.retrieve_from_memory(embedding)
    assert len(retrieved) > 0
