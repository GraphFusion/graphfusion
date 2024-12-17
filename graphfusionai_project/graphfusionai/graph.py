import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_knowledge(self, source, relation, target):
        self.graph.add_edge(source, target, relation=relation)

    def query_graph(self, source=None, relation=None, target=None):
        results = []
        for u, v, data in self.graph.edges(data=True):
            if (source is None or u == source) and \
               (target is None or v == target) and \
               (relation is None or data['relation'] == relation):
                results.append((u, data['relation'], v))
        return results

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

    def save_graph(self, path):
        nx.write_gpickle(self.graph, path)
        def load_graph(self, path):
            self.graph = nx.read_gpickle(path)
            