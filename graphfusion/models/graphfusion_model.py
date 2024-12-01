import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import networkx as nx
from core.memory_network import DynamicMemoryCell
from core.knowledge_graph import KnowledgeGraphEmbedder
from core.fusion_layer import FusionLayer

class GraphFusionAI(nn.Module):
    def __init__(self, config, knowledge_graph=None):
        super().__init__()
        
        # Use provided graph or create default
        if knowledge_graph is None:
            knowledge_graph = nx.Graph()
        
        # Configure model components
        self.config = config
        
        self.neural_memory = DynamicMemoryCell(
            input_dim=config.get('input_dim', 256),
            memory_dim=config.get('memory_dim', 512),
            context_dim=config.get('context_dim', 128)
        )
        
        # Create knowledge graph embedder
        self.knowledge_graph = KnowledgeGraphEmbedder.from_networkx(
            knowledge_graph, 
            embedding_dim=config.get('embedding_dim', 64)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(
                self.neural_memory.memory_dim + self.knowledge_graph.embedding_dim, 
                config.get('fusion_dim', 256)
            ),
            nn.ReLU(),
            nn.Linear(
                config.get('fusion_dim', 256), 
                config.get('output_dim', 128)
            )
        )
    
    def forward(self, input_context, previous_memory):
        # Neural memory processing
        neural_rep, _ = self.neural_memory(
            input_context, 
            previous_memory
        )
        
        # Get graph tensor representation
        heads, relations, tails = self.knowledge_graph.get_graph_tensor_representation()
        
        # Knowledge graph embedding
        graph_rep = self.knowledge_graph(
            heads=heads, 
            relations=relations, 
            tails=tails
        )
        
        # Combine representations
        combined_rep = torch.cat([neural_rep, graph_rep], dim=-1)
        
        # Fusion
        fused_representation = self.fusion_layer(combined_rep)
        
        return fused_representation
    
    def update_knowledge_graph(self, graph):
        """
        Dynamically update the knowledge graph
        """
        self.knowledge_graph = KnowledgeGraphEmbedder.from_networkx(
            graph, 
            embedding_dim=self.config.get('embedding_dim', 64)
        )
    
    def learn_graph_representation(self, training_graphs):
        """
        Learn representations from multiple graphs
        """
        # Merge graphs
        merged_graph = nx.compose_all(training_graphs)
        
        # Update knowledge graph
        self.update_knowledge_graph(merged_graph)
        
        return merged_graph

# Example usage
def create_example_model():
    # Create a sample knowledge graph
    G = nx.Graph()
    G.add_edge('Alice', 'Project1', relation='works_on')
    G.add_edge('Bob', 'Project1', relation='manages')
    
    # Model configuration
    config = {
        'input_dim': 256,
        'memory_dim': 512,
        'embedding_dim': 64,
        'output_dim': 128
    }
    
    # Initialize GraphFusionAI with the graph
    model = GraphFusionAI(config, knowledge_graph=G)
    
    return model

# Demonstration script
def main():
    # Create model
    model = create_example_model()
    
    # Generate sample input
    input_context = torch.randn(32, 256)
    previous_memory = torch.randn(32, 512)
    
    # Generate fused representation
    fused_rep = model(input_context, previous_memory)
    
    print("Fused Representation Shape:", fused_rep.shape)

if __name__ == "__main__":
    main()