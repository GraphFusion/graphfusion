from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RecommendationEngine:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
    
    def get_recommendations(self, query_embedding, top_k=5):
        """Retrieve and rank recommendations based on query embedding."""
        similar_entries = self.memory_manager.retrieve_similar(query_embedding, top_k)
        
        # Generate recommendations based on the similar entries
        recommendations = []
        for entry in similar_entries:
            similarity_score = cosine_similarity(query_embedding.reshape(1, -1), entry["embedding"].reshape(1, -1)).item()
            recommendations.append({
                "data": entry["data"],
                "label": entry.get("label", "No label"),
                "similarity_score": similarity_score
            })
        return recommendations

    def filter_by_context(self, recommendations, context):
        """Filter recommendations based on a specified context (e.g., diagnosis, treatment)."""
        filtered_recommendations = [
            recommendation for recommendation in recommendations
            if recommendation["label"] == context.get("label")
        ]
        return filtered_recommendations

    def adjust_confidence_based_on_feedback(self, recommendations, feedback):
        """Adjust the confidence score of recommendations based on user feedback."""
        adjustment_factor = 1.1 if feedback["relevance"] >= 0.5 else 0.9
        for recommendation in recommendations:
            recommendation["similarity_score"] *= adjustment_factor
        return recommendations

    def explain_recommendation(self, recommendation):
        """Generates a detailed explanation for the given recommendation."""
        explanation = (
            f"Recommended data: {recommendation['data']}\n"
            f"Label: {recommendation.get('label', 'No label')}\n"
            f"Similarity score: {recommendation['similarity_score']:.2f}\n"
            f"Contextual match: High relevance to query context."
        )
        return explanation

    def recommend_from_memory(self, context_label=None, top_k=5):
        """Recommend entries directly from memory, optionally filtered by context."""
        all_memory = self.memory_manager.memory
        if context_label:
            filtered_memory = [entry for entry in all_memory if entry["label"] == context_label]
        else:
            filtered_memory = all_memory
        
        # Sort by similarity score, if available
        sorted_recommendations = sorted(
            filtered_memory, key=lambda x: x.get("similarity_score", 0), reverse=True
        )[:top_k]
        
        # Prepare recommendations format
        recommendations = [
            {
                "data": entry["data"],
                "label": entry.get("label", "No label"),
                "similarity_score": entry.get("similarity_score", 0)
            }
            for entry in sorted_recommendations
        ]
        return recommendations
