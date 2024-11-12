import pytest
from graphfusion.nmn_core import NeuralMemoryNetwork

def test_embedding():
    nmn = NeuralMemoryNetwork()
    embedding = nmn.embed_text("Test clinical input")
    assert embedding.shape[1] > 0

def test_similarity():
    nmn = NeuralMemoryNetwork()
    emb1 = nmn.embed_text("clinical input")
    emb2 = nmn.embed_text("similar clinical input")
    assert nmn.compute_similarity(emb1, emb2).item() > 0.5
