# graphfusionai/utils.py
import networkx as nx
import json

class Utils:
    @staticmethod
    def validate_graph(graph):
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Input must be a networkx DiGraph")
        return True

    @staticmethod
    def export_graph(graph, file_path, format="json"):
        if format.lower() == "json":
            graph_data = {
                "nodes": list(graph.nodes(data=True)),
                "edges": list(graph.edges(data=True))
            }
            
            with open(file_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")