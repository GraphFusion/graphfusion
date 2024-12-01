import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphFusionLayer(nn.Module):
    def __init__(self, neural_memory, knowledge_graph):
        super().__init__()
        self.neural_memory = neural_memory
        self.knowledge_graph = knowledge_graph
        
        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.randn(2))
        
        # Projection layers
        self.memory_projection = nn.Linear(
            neural_memory.memory_dim, 
            knowledge_graph.embedding_dim
        )
        
        self.graph_projection = nn.Linear(
            knowledge_graph.embedding_dim, 
            neural_memory.memory_dim
        )
        
        # Attention mechanism for fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=knowledge_graph.embedding_dim,
            num_heads=4
        )
    
    def forward(self, input_context, previous_memory):
        # Neural memory update
        new_memory, _ = self.neural_memory(input_context, previous_memory)
        
        # Project memory to graph embedding space
        memory_embed = self.memory_projection(new_memory)
        
        # Knowledge graph reasoning
        # Assuming input_context contains entity indices
        graph_embed = self.knowledge_graph(
            heads=input_context, 
            relations=torch.zeros_like(input_context),
            tails=input_context
        )
        
        # Adaptive fusion with attention
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Attention-based fusion
        fused_embed, _ = self.fusion_attention(
            memory_embed.unsqueeze(0), 
            graph_embed.unsqueeze(0), 
            graph_embed.unsqueeze(0)
        )
        
        # Project back to memory space
        fused_memory = self.graph_projection(fused_embed.squeeze(0))
        
        return fused_memory