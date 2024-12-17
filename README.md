

</p>
<h1 align="center" weight='300'>GraphFusionAI: The Unified Opensource Framework for KnowledgeGraphs and Neural Memory Networks</h1>
<h3 align="center" weight='300'>Empowering AI with Persistent, Reliable, and Queryable Memory</h3>
<div align="center">

  [![GitHub release](https://img.shields.io/badge/Github-Release-blue)](https://github.com/GraphFusion/GraphFusion-NMN/releases)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/GraphFusion/GraphFusion/blob/main/LICENSE)
  [![Join us on Teams](https://img.shields.io/badge/Join-Teams-blue)](https://teams.microsoft.com/)
  [![Discord Server](https://img.shields.io/badge/Discord-Server-blue)](https://discord.gg/zK94WvRjZT)

</div>
<h3 align="center">
   <a href="https://github.com/GraphFusion/GraphFusion-NMN/blob/main/documentation.md"><b>Docs</b></a> &bull;
   <a href="https://graphfusion.github.io/graphfusion.io/"><b>Website</b></a>
</h3>
<br />

---


# **GraphFusionAI**

**GraphFusionAI** is a powerful library for integrating **dynamic neural memory** with **knowledge graphs**, enabling intelligent graph-based reasoning, contextual updates, and advanced querying for AI applications. The library is designed for flexibility, scalability, and ease of use across domains like conversational AI, healthcare, recommendation systems, and more.

## **Table of Contents**
- [Introduction](#introduction)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
- [Core Components](#core-components)  
  - [Graph](#graph)  
  - [Dynamic Memory Cell](#dynamic-memory-cell)  
  - [Knowledge Graph Embedding](#knowledge-graph-embedding)  
- [Example Use Cases](#example-use-cases)  
- [Documentation](#documentation)  
- [Contributing](#contributing)  
- [License](#license)  


## **Introduction**

GraphFusionAI combines two critical components of modern AI:  
1. **Knowledge Graphs** for structured representation of relationships between entities.  
2. **Dynamic Neural Memory** to enable context-aware and real-time updates using neural networks.

By fusing these components, the library allows for:  
- **Querying and reasoning** over complex knowledge graphs.  
- **Dynamic updates** with contextual embeddings.  
- **Seamless integration** with deep learning workflows.

This makes GraphFusionAI a robust tool for applications like conversational agents, knowledge-based systems, and more.

## **Key Features**

- **Graph Management**: Add, query, and manipulate nodes and edges with ease.
- **Dynamic Neural Memory**: Context-aware updates using GRU-based memory cells.
- **Graph Embeddings**: Compute embeddings for nodes, edges, and graphs.
- **Similarity Computation**: Measure semantic similarity between entities in a knowledge graph.
- **Visualization**: Easily visualize the structure of knowledge graphs.
- **Integration**: Compatible with tools like PyTorch and NetworkX for seamless workflows.
- **Scalable and Modular**: Designed for extensibility and scalability for larger graphs and real-world applications.

## **Installation**

To install the library, run the following command:

```bash
pip install graphfusionai
```

Ensure the following dependencies are installed in your environment:
- `torch`
- `networkx`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can also install directly from the source code:

```bash
git clone https://github.com/yourusername/graphfusionai.git
cd graphfusionai
pip install .
```

## **Quick Start**

Here is a simple example to get started with GraphFusionAI:

### **Graph Operations**
```python
from graphfusionai.graph import Graph

# Initialize the graph
graph = Graph()

# Add knowledge
graph.add_knowledge('Apple', 'is_fruit', 'Fruit')
graph.add_knowledge('Fruit', 'is_category', 'Food')

# Query the graph
results = graph.query_graph(source='Apple')
print(f"Query Results: {results}")

# Visualize the graph
graph.visualize()
```

**Output**:
```
Query Results: [('Apple', 'is_fruit', 'Fruit')]
```

### **Dynamic Memory Cell**
```python
from graphfusionai.memory import DynamicMemoryCell
import torch

# Initialize the memory cell
memory_cell = DynamicMemoryCell(input_dim=256, memory_dim=512, context_dim=128)

# Define input and memory tensors
input_tensor = torch.randn(256)
previous_memory = torch.randn(512)

# Update memory
updated_memory, attention_weights = memory_cell(input_tensor, previous_memory)
print("Updated Memory:", updated_memory)
print("Attention Weights:", attention_weights)
```

## **Core Components**

### **1. Graph**

The `Graph` class enables users to create and manipulate knowledge graphs:

- **Add knowledge**:
  ```python
  graph.add_knowledge('Entity1', 'relation', 'Entity2')
  ```

- **Query knowledge**:
  ```python
  graph.query_graph(source='Entity1', relation='relation')
  ```

- **Visualize**:
  ```python
  graph.visualize()
  ```

### **2. Dynamic Memory Cell**

The `DynamicMemoryCell` class models dynamic neural memory using GRU-based updates:

- **Initialize**:
  ```python
  memory_cell = DynamicMemoryCell(input_dim=256, memory_dim=512, context_dim=128)
  ```

- **Update memory**:
  ```python
  updated_memory, attention_weights = memory_cell(input_tensor, previous_memory)
  ```

### **3. Knowledge Graph Embedding**

The `KnowledgeGraphEmbedder` computes embeddings for nodes, relations, and graphs.

- **Compute Similarity**:
  ```python
  from graphfusionai.embedder import KnowledgeGraphEmbedder

  embedder = KnowledgeGraphEmbedder(graph)
  similarity = embedder.compute_graph_similarity('Entity1', 'Entity2')
  print(f"Similarity: {similarity}")
  ```

## **Example Use Cases**

### **1. Conversational AI**
Build conversational agents that can store and retrieve contextual knowledge dynamically.

```python
from graphfusionai.graph import Graph

graph = Graph()
graph.add_knowledge('User123', 'asked_about', 'Weather')
response = graph.query_graph(source='User123')
print(f"Response: {response}")
```

### **2. Healthcare**
Model and query patient medical histories with contextual updates.

```python
graph.add_knowledge('Patient123', 'diagnosed_with', 'Diabetes')
graph.add_knowledge('Patient123', 'prescribed', 'Metformin')
history = graph.query_graph(source='Patient123')
print(f"Patient History: {history}")
```

### **3. Recommendation Systems**
Leverage graph embeddings to compute similarities and recommend items.

```python
similarity = embedder.compute_graph_similarity('Product1', 'Product2')
print(f"Product Similarity: {similarity}")
```

## **Documentation**

For a detailed explanation of all methods, classes, and configurations, please refer to the [official documentation](https://github.com/yourusername/graphfusionai).

## **Contributing**

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation.

To run tests:
```bash
pytest tests/
```

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## **Contact**

For support or collaboration, feel free to reach out:  
**Email**: [hello@GraphFusion.onmicrosoft.com](mailto:hello@GraphFusion.onmicrosoft.com)  
**GitHub**: [https://github.com/GraphFusion/graphfusion](https://github.com/GraphFusion/graphfusion)  

Join our community on Discord: [GraphFusionAI Community](https://discord.gg/f9HtUN2x)
