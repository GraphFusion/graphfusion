import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
from graphfusionai.memory import DynamicMemoryCell

def test_memory_cell_initialization():
    cell = DynamicMemoryCell(input_dim=8, memory_dim=8, context_dim=8)
    assert cell.input_projection is not None

def test_memory_update():
    cell = DynamicMemoryCell(input_dim=8, memory_dim=8, context_dim=8)
    input_tensor = torch.randn(8)
    previous_memory = torch.zeros(8)
    updated_memory, _ = cell(input_tensor, previous_memory)
    assert updated_memory.shape == (8,)
