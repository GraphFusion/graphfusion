from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
    
    def get_recommendations(self, query_embedding, top_k=5):
        """Retrieve and rank recommendations based on query embedding."""
        
        # Retrieve top_k most similar entries from memory
        similar_entries = self.memory_manager.retrieve_similar(query_embedding, top_k)
        
        # Prepare the recommendations with actual similarity scores
        recommendations = []
        for entry in similar_entries:
            similarity_score = cosine_similarity(
                query_embedding.reshape(1, -1),
                entry["embedding"].reshape(1, -1)
            ).item()  # Calculate similarity score
            
            recommendations.append({
                "data": entry["data"],
                "label": entry.get("label", "No label"),
                "similarity_score": similarity_score  # Use actual similarity score
            })
        
        return recommendations
