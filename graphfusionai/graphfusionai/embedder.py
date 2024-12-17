import torch
import networkx as nx

class KnowledgeGraphEmbedder:
    def __init__(self, graph, embedding_dim=64):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.node_embeddings = {}
        for node in graph.nodes():
            self.node_embeddings[node] = torch.nn.Parameter(torch.randn(embedding_dim))
        unique_relations = set(nx.get_edge_attributes(graph, 'relation').values())
        self.relation_embeddings = {rel: torch.nn.Parameter(torch.randn(embedding_dim)) for rel in unique_relations}

    def get_node_embedding(self, node):
        if node not in self.node_embeddings:
            self.node_embeddings[node] = torch.nn.Parameter(torch.randn(self.embedding_dim))
        return self.node_embeddings[node]

    def compute_similarity(self, node1, node2):
        emb1 = self.get_node_embedding(node1)
        emb2 = self.get_node_embedding(node2)
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)