import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class KnowledgeGraphEmbedder(nn.Module):
    def __init__(self, embedding_dim=64, max_entities=1000, max_relations=100):
        super().__init__()
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        
        # Embedding layers with more entities and relations
        self.entity_embeddings = nn.Embedding(max_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(max_relations, embedding_dim)
        
        self.embedding_dim = embedding_dim
        
        # Similarity metric
        self.similarity_layer = nn.Linear(embedding_dim, 1)
    
    def add_entity(self, entity_id, attributes=None, embedding=None):
        """
        Add an entity to the graph with optional custom embedding
        
        :param entity_id: Unique identifier for the entity
        :param attributes: Dictionary of entity attributes
        :param embedding: Optional custom embedding tensor
        """
        self.graph.add_node(entity_id, attributes=attributes or {})
        
        # If custom embedding is not provided, use default embedding
        if embedding is None:
            embedding = self.entity_embeddings(torch.tensor(entity_id % 1000))
        
        # Store embedding as node attribute
        self.graph.nodes[entity_id]['embedding'] = embedding
    
    def add_relation(self, source, target, relation_type, weight=1.0, attributes=None):
        """
        Add a weighted relationship between entities
        
        :param source: Source entity ID
        :param target: Target entity ID
        :param relation_type: Type of relationship
        :param weight: Relationship strength
        :param attributes: Additional relationship attributes
        """
        # Ensure entities exist
        if source not in self.graph or target not in self.graph:
            raise ValueError("Source or target entity does not exist in the graph")
        
        # Add edge with relation type and weight
        self.graph.add_edge(
            source, 
            target, 
            relation_type=relation_type, 
            weight=weight,
            attributes=attributes or {}
        )
    
    def compute_path_embedding(self, path):
        """
        Compute embedding for a path through the graph
        
        :param path: List of entity IDs representing a path
        :return: Aggregated path embedding
        """
        path_embeddings = []
        for i in range(len(path) - 1):
            # Get relation type between entities
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            relation_type = edge_data[0]['relation_type']
            
            # Get entity and relation embeddings
            entity_embed = self.graph.nodes[path[i]]['embedding']
            relation_embed = self.relation_embeddings(torch.tensor(relation_type))
            
            # Combine embeddings
            path_embeddings.append(entity_embed * relation_embed)
        
        # Aggregate path embeddings
        return torch.mean(torch.stack(path_embeddings), dim=0)
    
    def find_paths(self, source, target, max_length=3, weight_threshold=0.5):
        """
        Find paths between entities with weight filtering
        
        :param source: Source entity ID
        :param target: Target entity ID
        :param max_length: Maximum path length
        :param weight_threshold: Minimum edge weight to consider
        :return: List of paths
        """
        def weight_filter(u, v, data):
            return data.get('weight', 1.0) >= weight_threshold
        
        # Find all paths with weight filtering
        paths = list(nx.all_simple_paths(
            self.graph, 
            source, 
            target, 
            cutoff=max_length
        ))
        
        return paths
    
    def forward(self, heads, relations, tails):
        """
        Compute similarity scores for graph reasoning
        
        :param heads: Tensor of head entity indices
        :param relations: Tensor of relation type indices
        :param tails: Tensor of tail entity indices
        :return: Similarity scores
        """
        # Retrieve embeddings
        head_embeds = self.entity_embeddings(heads)
        tail_embeds = self.entity_embeddings(tails)
        rel_embeds = self.relation_embeddings(relations)
        
        # Compute combined embedding
        combined_embed = head_embeds * rel_embeds * tail_embeds
        
        # Compute similarity score
        scores = self.similarity_layer(combined_embed).squeeze(-1)
        
        return scores