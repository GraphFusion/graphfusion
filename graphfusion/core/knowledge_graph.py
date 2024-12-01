import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class KnowledgeGraphEmbedder(nn.Module):
    def __init__(self, graph: nx.Graph, embedding_dim: int = 64):
        super().__init__()
        
        # Extract unique entities and relations
        self.entities = list(graph.nodes())
        self.relations = list(set(nx.get_edge_attributes(graph, 'relation').values()))
        
        # Create mappings
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(self.relations)}
        
        # Embedding layers
        self.entity_embeddings_real = nn.Embedding(len(self.entities), embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(len(self.entities), embedding_dim)
        
        self.relation_embeddings_real = nn.Embedding(len(self.relations), embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(len(self.relations), embedding_dim)
        
        # Graph structure
        self.graph = graph
        self.embedding_dim = embedding_dim
    
    def get_graph_tensor_representation(self):
        """
        Convert NetworkX graph to tensor representation
        """
        edges = list(self.graph.edges(data=True))
        
        head_indices = torch.tensor([self.entity_to_idx[edge[0]] for edge in edges])
        tail_indices = torch.tensor([self.entity_to_idx[edge[1]] for edge in edges])
        relation_indices = torch.tensor([
            self.relation_to_idx[edge[2].get('relation', 'default')] 
            for edge in edges
        ])
        
        return head_indices, relation_indices, tail_indices
    
    def complex_scoring(self, 
                        head_real: torch.Tensor, 
                        head_imag: torch.Tensor,
                        rel_real: torch.Tensor, 
                        rel_imag: torch.Tensor,
                        tail_real: torch.Tensor, 
                        tail_imag: torch.Tensor) -> torch.Tensor:
        """
        ComplEx scoring function for knowledge graph embeddings
        """
        real_score = (
            head_real * rel_real * tail_real +
            head_imag * rel_imag * tail_real +
            head_real * rel_imag * tail_imag -
            head_imag * rel_real * tail_imag
        )
        
        return torch.sum(real_score, dim=-1)
    
    def forward(self, 
                heads: torch.Tensor, 
                relations: torch.Tensor, 
                tails: torch.Tensor):
        # Retrieve embeddings
        head_real = self.entity_embeddings_real(heads)
        head_imag = self.entity_embeddings_imag(heads)
        
        tail_real = self.entity_embeddings_real(tails)
        tail_imag = self.entity_embeddings_imag(tails)
        
        rel_real = self.relation_embeddings_real(relations)
        rel_imag = self.relation_embeddings_imag(relations)
        
        # Compute complex scoring
        scores = self.complex_scoring(
            head_real, head_imag, 
            rel_real, rel_imag, 
            tail_real, tail_imag
        )
        
        return scores
    
    @classmethod
    def from_networkx(cls, graph: nx.Graph, embedding_dim: int = 64):
        """
        Class method to create embedder directly from NetworkX graph
        """
        return cls(graph, embedding_dim)

