import os
import sys
from graphfusion.memory_manager import MemoryManager
from graphfusion.recommendation_engine import RecommendationEngine
from graphfusion.knowledge_graph import KnowledgeGraph
from transformers import AutoModel, AutoTokenizer
import torch

class NeuralMemoryNetwork:
    def __init__(self, memory_manager=None, recommendation_engine=None, knowledge_graph=None):
        # Initialize components if they are not passed
        self.memory_manager = memory_manager or MemoryManager()
        self.recommendation_engine = recommendation_engine or RecommendationEngine(self.memory_manager)  # Ensure MemoryManager is passed here
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

        # Initialize embedding model
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def generate_embedding(self, text):
        """Convert text to embedding."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1)
        return outputs.numpy()

    def store_in_memory(self, text, data, label=None):
        """Store data in memory and update knowledge graph."""
        embedding = self.generate_embedding(text)
        node_id = self.knowledge_graph.add_node(data, label=label)
        
        # Store embedding with the node reference in memory
        self.memory_manager.store_in_memory(embedding, data, label)
        
        # Update KG relationships based on similarity with existing nodes
        similar_nodes = self.knowledge_graph.find_similar_nodes(embedding, threshold=0.8)
        for similar_node in similar_nodes:
            self.knowledge_graph.add_edge(node_id, similar_node["node_id"], relation="similar_to")
    
    def retrieve_similar(self, text, top_k=5):
        """Retrieve similar cases using both memory and knowledge graph."""
        query_embedding = self.generate_embedding(text)
        similar_cases = self.memory_manager.retrieve_similar(query_embedding, top_k)
        
        # Use KG to retrieve contextually linked cases
        kg_recommendations = self.knowledge_graph.retrieve_linked_nodes(similar_cases)
        return similar_cases + kg_recommendations

    def generate_recommendations(self, text, top_k=5):
        """Generate recommendations based on similar cases and KG."""
        similar_cases = self.retrieve_similar(text, top_k)
        return self.recommendation_engine.generate_recommendations(similar_cases)

    def update_on_feedback(self, feedback):
        """Update memory and knowledge graph based on feedback."""
        self.memory_manager.update_on_feedback(feedback)
        
        # Adjust relevance of connections in the KG based on feedback
        for entry in feedback:
            if entry["relevance"] < 0.5:
                self.knowledge_graph.reduce_edge_weight(entry["node_id"])

