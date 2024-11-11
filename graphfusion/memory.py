import networkx as nx

class Memory:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for storing relationships

    def store(self, entity, relation, value):
        """Store a relationship between two entities."""
        self.graph.add_edge(entity, relation, value=value)

    def retrieve(self, entity, relation):
        """Retrieve a relationship from the graph."""
        for _, _, data in self.graph.edges(entity, data=True):
            if data.get("value") == relation:
                return data
        return None

    def query(self, entity):
        """Query all relationships of an entity."""
        return list(self.graph.neighbors(entity))
