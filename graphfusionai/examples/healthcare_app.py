from graphfusionai.graph import Graph

# Initialize graph
patient_graph = Graph()

# Add medical history
patient_graph.add_knowledge('Patient123', 'diagnosed_with', 'Hypertension')
patient_graph.add_knowledge('Patient123', 'prescribed', 'MedicationX')

# Query history
history = patient_graph.query_graph(source='Patient123')
print(f"Patient History: {history}")
