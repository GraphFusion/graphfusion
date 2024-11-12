# Importing NMN and initializing with a specific model
from graphfusion.nmn_core import NeuralMemoryNetwork

# Initialize the NMN
nmn = NeuralMemoryNetwork(model_name="bert-base-uncased")

# Process clinical text data
patient_data = "Patient presents with fever and cough, has a history of asthma."
embedding = nmn.embed_text(patient_data)

# Compare similarity between two clinical records
record_1 = "Patient complains of shortness of breath, fever."
record_2 = "Patient has asthma and reported breathing difficulties."
embedding1 = nmn.embed_text(record_1)
embedding2 = nmn.embed_text(record_2)

similarity_score = nmn.compute_similarity(embedding1, embedding2)
print("Similarity Score:", similarity_score.item())
