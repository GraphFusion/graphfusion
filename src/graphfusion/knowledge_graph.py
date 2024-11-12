import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id, attributes=None):
        """Adds a new node to the graph with optional attributes."""
        self.graph.add_node(node_id, **(attributes or {}))
    
    def add_edge(self, source_node, target_node, relationship_type, confidence=1.0):
        """Adds a directed edge between source and target with a relationship type and confidence."""
        self.graph.add_edge(source_node, target_node, type=relationship_type, confidence=confidence)
    
    def update_edge(self, source_node, target_node, new_confidence):
        """Updates the confidence score of an existing edge."""
        if self.graph.has_edge(source_node, target_node):
            self.graph[source_node][target_node]["confidence"] = new_confidence
    
    def get_related_nodes(self, node_id, relationship_type=None):
        """Returns nodes directly related to the given node, optionally filtered by relationship type."""
        related = []
        for target in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, target)
            if relationship_type is None or edge_data["type"] == relationship_type:
                related.append((target, edge_data))
        return related
    
    def find_path(self, node_start, node_end):
        """Finds a path from node_start to node_end if it exists."""
        try:
            return nx.shortest_path(self.graph, source=node_start, target=node_end)
        except nx.NetworkXNoPath:
            return None
    
    def find_cluster(self, node_id, radius=2):
        """Finds a cluster of nodes within a given radius (degree) from the node_id."""
        return list(nx.single_source_shortest_path_length(self.graph, node_id, cutoff=radius).keys())
    
    def recommend_based_on_graph(self, query_node, top_k=5):
        """Recommends related nodes based on confidence scores."""
        related_nodes = self.get_related_nodes(query_node)
        # Sort by confidence and return top_k results
        sorted_nodes = sorted(related_nodes, key=lambda x: x[1]["confidence"], reverse=True)
        return sorted_nodes[:top_k]
