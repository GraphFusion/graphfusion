import torch
import torch.nn as nn
import pandas as pd
from ..core.memory_network import DynamicMemoryCell
from ..core.knowledge_graph import KnowledgeGraphEmbedder
from ..core.fusion_layer import GraphFusionLayer
from ..utils.data_loader import GraphFusionDataset

class GraphFusionAI(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # Default configuration
        self.config = config or {
            'input_dim': 256,
            'memory_dim': 512,
            'context_dim': 128,
            'embedding_dim': 64,
            'max_entities': 1000,
            'max_relations': 100
        }
        
        # Initialize components
        self.neural_memory = DynamicMemoryCell(
            input_dim=self.config['input_dim'],
            memory_dim=self.config['memory_dim'],
            context_dim=self.config['context_dim']
        )
        
        self.knowledge_graph = KnowledgeGraphEmbedder(
            embedding_dim=self.config['embedding_dim'],
            max_entities=self.config['max_entities'],
            max_relations=self.config['max_relations']
        )
        
        self.fusion_layer = GraphFusionLayer(
            self.neural_memory, 
            self.knowledge_graph
        )
    
    def build_knowledge_graph(self, entities_path, relations_path):
        """
        Construct knowledge graph from CSV data
        
        :param entities_path: Path to entities CSV
        :param relations_path: Path to relations CSV
        :return: Populated GraphFusionDataset
        """
        # Load entities
        entities_df = pd.read_csv(entities_path)
        for _, row in entities_df.iterrows():
            # Create custom embedding if needed
            entity_embedding = None
            if 'embedding' in row:
                entity_embedding = torch.tensor(
                    eval(row['embedding']), 
                    dtype=torch.float32
                )
            
            self.knowledge_graph.add_entity(
                row['entity_id'], 
                attributes=row.to_dict(),
                embedding=entity_embedding
            )
        
        # Load relations
        relations_df = pd.read_csv(relations_path)
        for _, row in relations_df.iterrows():
            self.knowledge_graph.add_relation(
                row['source'], 
                row['target'], 
                row['relation_type'],
                weight=row.get('weight', 1.0),
                attributes=row.to_dict()
            )
        
        # Create and return dataset
        return GraphFusionDataset(self.knowledge_graph)
    
    def forward(self, input_context, previous_memory):
        """
        Forward pass through the GraphFusionAI model
        
        :param input_context: Input context tensor
        :param previous_memory: Previous memory state
        :return: Fused representation
        """
        return self.fusion_layer(input_context, previous_memory)
    
    def reason_over_graph(self, source_entity, target_entity, max_path_length=3):
        """
        Perform reasoning over the knowledge graph
        
        :param source_entity: Starting entity
        :param target_entity: Target entity
        :param max_path_length: Maximum path length to explore
        :return: List of possible paths with their embeddings
        """
        # Find paths
        paths = self.knowledge_graph.find_paths(
            source_entity, 
            target_entity, 
            max_length=max_path_length
        )
        
        # Compute path embeddings
        path_embeddings = [
            (path, self.knowledge_graph.compute_path_embedding(path))
            for path in paths
        ]
        
        return path_embeddings