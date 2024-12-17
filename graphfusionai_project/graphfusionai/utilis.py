import networkx as nx
import json

class Utils:
    @staticmethod
    def validate_graph(graph):
        if not isinstance(graph, nx.Graph):
            raise ValueError("Input must be a NetworkX graph.")

    @staticmethod
    def export_graph(graph, filename, format="json"):
        if format == "json":
            with open(filename, 'w') as f:
                json.dump(nx.node_link_data(graph), f)
        elif format == "graphml":
            nx.write_graphml(graph, filename)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'graphml'.")