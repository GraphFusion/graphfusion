import pytest
import torch
from transformers import AutoModel, AutoTokenizer
from graphfusion.nmn_core import NeuralMemoryNetwork  # Make sure the import path is correct

@pytest.fixture
def setup_nmn():
    """
    Fixture to set up the NeuralMemoryNetwork for testing.
    This will be run before each test that uses this fixture.
    """
    nmn = NeuralMemoryNetwork()  # Initialize the NeuralMemoryNetwork
    return nmn

def test_embed_text(setup_nmn):
    """
    Test the embedding generation for a simple text.
    This will check if the embedding is generated correctly and has the expected shape.
    """
    nmn = setup_nmn
    text = "Hello, this is a test sentence."
    embedding = nmn.embed_text(text)
    
    # Check if the output is a tensor and has the correct shape
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, nmn.model.config.hidden_size)

def test_compute_similarity(setup_nmn):
    """
    Test the similarity computation between two text embeddings.
    This will ensure that the similarity calculation works as expected.
    """
    nmn = setup_nmn
    text1 = "Hello, this is a test sentence."
    text2 = "Hello, this is another test sentence."
    
    embedding1 = nmn.embed_text(text1)
    embedding2 = nmn.embed_text(text2)
    
    similarity = nmn.compute_similarity(embedding1, embedding2)
    
    # Check if similarity is a tensor with shape torch.Size([1])
    assert isinstance(similarity, torch.Tensor)
    assert similarity.shape == torch.Size([1])  # Expect a tensor with one element

def test_analyze_input(setup_nmn):
    """
    Test the analysis of input text.
    This test ensures the function returns both an embedding and the original text.
    """
    nmn = setup_nmn
    text = "This is an analysis test."
    
    result = nmn.analyze_input(text)
    
    # Check if the result contains 'embedding' and 'text'
    assert 'embedding' in result
    assert 'text' in result
    
    # Check that the embedding is a tensor
    assert isinstance(result['embedding'], torch.Tensor)
    assert result['embedding'].shape == (1, nmn.model.config.hidden_size)
    
    # Check if the text is correct
    assert result['text'] == text

def test_embedding_consistency(setup_nmn):
    """
    Test that embedding for the same text is consistent.
    This ensures that embeddings for the same text are always identical.
    """
    nmn = setup_nmn
    text = "Consistency test."
    
    embedding1 = nmn.embed_text(text)
    embedding2 = nmn.embed_text(text)
    
    # Check if embeddings for the same text are identical
    assert torch.equal(embedding1, embedding2), "Embeddings should be identical for the same input text."

def test_similarity_with_identical_text(setup_nmn):
    """
    Test that similarity between identical text is close to 1.
    This ensures that the model considers identical texts to be highly similar.
    """
    nmn = setup_nmn
    text = "Same text similarity."
    
    embedding1 = nmn.embed_text(text)
    embedding2 = nmn.embed_text(text)
    
    similarity = nmn.compute_similarity(embedding1, embedding2)
    
    # Check if similarity is close to 1
    assert similarity.item() > 0.9, f"Similarity is too low: {similarity.item()}"

def test_empty_input(setup_nmn):
    """
    Test how the model handles empty input.
    This ensures that an empty string doesn't cause errors.
    """
    nmn = setup_nmn
    text = ""
    
    result = nmn.analyze_input(text)
    
    # Check that the embedding shape is as expected even for empty text
    assert result["embedding"].shape == (1, nmn.model.config.hidden_size)
    assert result["text"] == text
