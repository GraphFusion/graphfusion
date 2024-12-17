import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
import networkx as nx
from graphfusionai.embedder import KnowledgeGraphEmbedder

def test_initialize_embeddings():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", relation="related_to")
    embedder = KnowledgeGraphEmbedder(graph, embedding_dim=16)
    assert "A" in embedder.node_embeddings
    assert "B" in embedder.node_embeddings

def test_get_node_embedding():
    graph = nx.DiGraph()
    graph.add_node("A")
    embedder = KnowledgeGraphEmbedder(graph, embedding_dim=16)
    embedding = embedder.get_node_embedding("A")
    assert embedding.shape == (16,)

def test_compute_similarity():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", relation="related_to")
    embedder = KnowledgeGraphEmbedder(graph, embedding_dim=16)
    similarity = embedder.compute_similarity("A", "B")
    assert -1 <= similarity.item() <= 1
