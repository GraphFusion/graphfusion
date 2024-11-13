class RecommendationEngine:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
    
    def get_recommendations(self, query_embedding, top_k=5):
        """Retrieve and rank recommendations based on query embedding."""
        similar_entries = self.memory_manager.retrieve_similar(query_embedding, top_k)
        
        # Generate recommendations based on the similar entries.
        recommendations = []
        for entry in similar_entries:
            recommendations.append({
                "data": entry["data"],
                "label": entry.get("label", "No label"),
                "similarity_score": entry["embedding"]
            })
        return recommendations
