import networkx as nx
import matplotlib.pyplot as plt
import torch

from graphfusionai.memory import DynamicMemoryCell
from graphfusionai.embedder import KnowledgeGraphEmbedder

class GraphFusionAI:
    def __init__(self, input_dim=256, memory_dim=512, context_dim=128):
        """
        Initialize GraphFusionAI with core components

        Args:
            input_dim (int): Dimension of input tensors
            memory_dim (int): Dimension of memory cells
            context_dim (int): Dimension of context representation
        """
        self.graph = nx.DiGraph()

        self.memory_cell = DynamicMemoryCell(
            input_dim=input_dim,
            memory_dim=memory_dim,
            context_dim=context_dim
        )

        self.graph_embedder = KnowledgeGraphEmbedder(
            graph=self.graph,
            embedding_dim=64
        )

    def add_knowledge(self, source, relation, target):
        """
        Add knowledge to the graph

        Args:
            source: Source node
            relation: Relationship between nodes
            target: Target node
        """
        self.graph.add_edge(source, target, relation=relation)
        
        # Update graph embedder
        if source not in self.graph_embedder.node_embeddings:
            self.graph_embedder.node_embeddings[source] = torch.nn.Parameter(torch.randn(self.graph_embedder.embedding_dim))
        if target not in self.graph_embedder.node_embeddings:
            self.graph_embedder.node_embeddings[target] = torch.nn.Parameter(torch.randn(self.graph_embedder.embedding_dim))

    def visualize_graph(self, save_path=None):
        """
        Visualize the current knowledge graph

        Args:
            save_path (str, optional): Path to save the graph visualization
        """
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=10, font_weight='bold')

        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("GraphFusionAI Knowledge Graph")
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def query_graph(self, source=None, relation=None, target=None):
        """
        Query the knowledge graph with flexible matching

        Args:
            source: Source node to match
            relation: Relationship to match
            target: Target node to match

        Returns:
            List of tuples containing matching graph edges
        """
        results = []
        for u, v, data in self.graph.edges(data=True):
            if (source is None or u == source) and \
               (target is None or v == target) and \
               (relation is None or data['relation'] == relation):
                results.append((u, data['relation'], v))
        return results