from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MemoryManager:
    def __init__(self):
        # Initialize storage for embeddings and associated data
        self.memory = []
        
    def store_in_memory(self, embedding, data, label=None):
        """Stores a new data entry with its embedding and optional label."""
        self.memory.append({
            "embedding": embedding,
            "data": data,
            "label": label
        })
        
    def retrieve_similar(self, query_embedding, top_k=5):
        """Retrieves top_k most similar entries based on cosine similarity."""
        if not self.memory:
            return []
        
        similarities = [
            cosine_similarity(query_embedding.reshape(1, -1), entry["embedding"].reshape(1, -1)).item()
            for entry in self.memory
        ]
        
        # Sort memory items by similarity score in descending order
        sorted_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memory[i] for i in sorted_indices]
    
    def update_entry(self, memory_id, new_data):
        """Updates an existing memory entry with new data."""
        if 0 <= memory_id < len(self.memory):
            self.memory[memory_id]["data"] = new_data
    
    def forget_old_entries(self, threshold=0.2):
        """Removes entries with relevance below a certain threshold."""
        self.memory = [
            entry for entry in self.memory
            if cosine_similarity(entry["embedding"].reshape(1, -1), entry["embedding"].reshape(1, -1)).item() > threshold
        ]
    
    def update_on_feedback(self, feedback):
        """Adjusts memory relevance based on feedback (e.g., boosts or reduces similarity scores)."""
        for entry in self.memory:
            if feedback["relevance"] < 0.5:  # Example condition based on feedback
                entry["embedding"] *= 0.9  # Decrease relevance of similar entries
