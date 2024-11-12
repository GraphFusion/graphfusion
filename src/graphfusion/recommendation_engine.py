class RecommendationEngine:
    def __init__(self, knowledge_graph, memory_manager):
        self.knowledge_graph = knowledge_graph
        self.memory_manager = memory_manager

    def get_recommendations(self, query_node, top_k=5):
        """Retrieve and rank recommendations based on query node."""
        # Retrieve related nodes and their confidence scores
        related_nodes = self.knowledge_graph.get_related_nodes(query_node)
        
        # Sort by confidence score
        ranked_recommendations = sorted(
            related_nodes, key=lambda x: x[1]["confidence"], reverse=True
        )
        return ranked_recommendations[:top_k]
    
    def filter_by_context(self, recommendations, context):
        """Filter recommendations based on contextual information."""
        filtered = [
            rec for rec in recommendations if self._matches_context(rec, context)
        ]
        return filtered
    
    def adjust_confidence_based_on_feedback(self, source, target, feedback):
        """Update confidence scores based on user feedback."""
        current_confidence = self.knowledge_graph.get_confidence(source, target)
        adjusted_confidence = current_confidence + feedback  # simple adjustment logic
        self.knowledge_graph.update_edge(source, target, adjusted_confidence)

    def explain_recommendation(self, recommendation):
        """Generate an explanation for a given recommendation."""
        source, edge_data = recommendation
        reason = f"Based on a confidence score of {edge_data['confidence']}, " \
                 f"the system recommends {source} due to its relation to the query."
        return reason
