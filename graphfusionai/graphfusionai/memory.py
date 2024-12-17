import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMemoryCell(nn.Module):
    def __init__(self, input_dim, memory_dim, context_dim):
        super().__init__()
        self.context_layer = nn.MultiheadAttention(embed_dim=context_dim, num_heads=4)
        self.memory_update = nn.GRUCell(input_size=input_dim, hidden_size=memory_dim)
        self.input_projection = nn.Linear(input_dim, context_dim)
        self.memory_projection = nn.Linear(memory_dim, context_dim)

    def forward(self, input_tensor, previous_memory, context=None):
        projected_input = self.input_projection(input_tensor)
        projected_memory = self.memory_projection(previous_memory)

        if context is None:
            context = projected_input

        context_aware_input, attention_weights = self.context_layer(
            projected_input.unsqueeze(0),
            projected_memory.unsqueeze(0),
            context.unsqueeze(0)
        )

        updated_memory = self.memory_update(context_aware_input.squeeze(0), previous_memory)
        return updated_memory, attention_weights