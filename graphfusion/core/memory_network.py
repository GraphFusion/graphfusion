import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMemoryCell(nn.Module):
    def __init__(self, input_dim, memory_dim, context_dim):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.context_dim = context_dim
        
        # Attention mechanism
        self.query_layer = nn.Linear(input_dim, context_dim)
        self.key_layer = nn.Linear(memory_dim, context_dim)
        self.value_layer = nn.Linear(memory_dim, memory_dim)
        
        # Memory update gates
        self.update_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        self.reset_gate = nn.Linear(input_dim + memory_dim, memory_dim)
        
        # Candidate memory
        self.candidate_layer = nn.Linear(input_dim + memory_dim, memory_dim)
        
    def forward(self, input_tensor, previous_memory):
        # Prepare query, key, and value
        query = self.query_layer(input_tensor)
        key = self.key_layer(previous_memory)
        value = self.value_layer(previous_memory)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Context-aware representation
        context_vector = torch.matmul(attention_probs, value)
        
        # Concatenate input and previous memory
        combined = torch.cat([input_tensor, previous_memory], dim=-1)
        
        # Compute gates
        update_gate = torch.sigmoid(self.update_gate(combined))
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        
        # Candidate memory
        candidate_memory = torch.tanh(
            self.candidate_layer(
                torch.cat([input_tensor, reset_gate * previous_memory], dim=-1)
            )
        )
        
        # Update memory
        new_memory = (1 - update_gate) * previous_memory + update_gate * candidate_memory
        
        return new_memory, attention_probs