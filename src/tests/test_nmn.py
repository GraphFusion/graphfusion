# test_nmn.py

from graphfusion.nmn_core import NeuralMemoryNetwork  # Adjust if your module path is different
import numpy as np

def test_nmn():
    # Initialize the NeuralMemoryNetwork
    nmn = NeuralMemoryNetwork()

    # Test: Storing data in memory and generating an embedding
    text = "This is a test sentence for embedding."
    data = {"patient_id": 123, "symptoms": "fever, cough"}  # Example of data that might be stored
    label = "Case A"

    # Store in memory
    nmn.store_in_memory(text, data, label)

    print("Data stored successfully in memory.")

    # Test: Retrieving similar cases
    query_text = "Fever and cough in patient"
    similar_cases = nmn.retrieve_similar(query_text, top_k=3)

    print(f"Similar cases for '{query_text}':")
    for case in similar_cases:
        print(case)

    # Test: Generating recommendations
    recommendations = nmn.generate_recommendations(query_text, top_k=3)
    
    print(f"Recommendations for '{query_text}':")
    for recommendation in recommendations:
        print(recommendation)

    # Test: Updating with feedback
    feedback = [{"node_id": case["node_id"], "relevance": 0.4} for case in similar_cases]  # Example feedback
    nmn.update_on_feedback(feedback)

    print("Feedback applied and memory/knowledge graph updated.")

if __name__ == "__main__":
    test_nmn()
