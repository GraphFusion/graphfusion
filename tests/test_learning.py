import unittest
from graphfusion import Learning
from graphfusion import Memory

class TestLearning(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.learning = Learning(self.memory)

    def test_update_model(self):
        self.learning.update_model([[1, 2], [3, 4]], [0, 1])
        prediction = self.learning.predict([[1, 2]])
        self.assertIn(prediction, [0, 1])
