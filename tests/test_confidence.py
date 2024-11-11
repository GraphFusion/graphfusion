import unittest
from graphfusion import Confidence
from graphfusion import Learning
from graphfusion import Memory

class TestConfidence(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.learning = Learning(self.memory)
        self.confidence = Confidence(self.learning, self.memory)

    def test_confidence_score(self):
        self.memory.store("patient123", "has_diagnosis", "Diabetes")
        self.learning.update_model([[1, 2], [3, 4]], [0, 1])
        score = self.confidence.score([[1, 2]])
        self.assertGreater(score, 0.5)
