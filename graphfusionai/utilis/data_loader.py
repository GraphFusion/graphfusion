import torch
from torch.utils.data import Dataset

class GraphFusionDataset(Dataset):
    def __init__(self, knowledge_graph):
        """
        Dataset wrapper for knowledge graph data
        
        :param knowledge_graph: KnowledgeGraphEmbedder instance
        """
        self.graph = knowledge_graph.graph
        self.knowledge_graph = knowledge_graph
        
        # Preprocess nodes and embeddings
        self.nodes = list(self.graph.nodes)
        self.node_embeddings = [
            self.graph.nodes[node]['embedding'] 
            for node in self.nodes
        ]
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        node = self.nodes[idx]
        embedding = self.node_embeddings[idx]
        
        # Get node neighbors and their relations
        neighbors = list(self.graph.neighbors(node))
        neighbor_embeddings = [
            self.graph.nodes[neighbor]['embedding'] 
            for neighbor in neighbors
        ]
        
        return {
            'node': node,
            'embedding': embedding,
            'neighbors': neighbors,
            'neighbor_embeddings': neighbor_embeddings
        }
    
    def get_subgraph(self, nodes):
        """
        Extract a subgraph for the given nodes
        
        :param nodes: List of node IDs
        :return: Subgraph
        """
        return self.graph.subgraph(nodes)