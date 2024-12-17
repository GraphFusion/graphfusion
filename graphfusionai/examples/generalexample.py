from graphfusionai.graph import Graph
from graphfusionai.memory import DynamicMemoryCell
from graphfusionai.embedder import KnowledgeGraphEmbedder
import torch


graph = Graph()
graph.add_knowledge('Apple', 'is_fruit', 'Fruit')
graph.query_graph(relation='is_fruit')
graph.visualize()


memory_cell = DynamicMemoryCell(input_dim=256, memory_dim=512, context_dim=128)
input_tensor = torch.randn(256)
previous_memory = torch.randn(512)
updated_memory, attention_weights = memory_cell(input_tensor, previous_memory)
embedder = KnowledgeGraphEmbedder(graph)
similarity = embedder.compute_graph_similarity('Apple', 'Banana')
print(similarity)