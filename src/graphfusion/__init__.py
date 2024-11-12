# src/graphfusion/__init__.py

from .nmn_core import NeuralMemoryNetwork
from .memory_manager import MemoryManager
from .knowledge_graph import KnowledgeGraph
from .recommendation_engine import RecommendationEngine

__version__ = "0.1.0"

__all__ = ["NeuralMemoryNetwork", "MemoryManager", "KnowledgeGraph", "RecommendationEngine"]
