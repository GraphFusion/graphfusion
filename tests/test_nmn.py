import unittest
from graphfusion import NeuralMemoryNetwork

class TestNMN(unittest.TestCase):
    def setUp(self):
        self.nmn = NeuralMemoryNetwork()

    def test_store_and_retrieve(self):
        self.nmn.store_data("patient123", "has_diagnosis", "Diabetes")
        result = self.nmn.retrieve_data("patient123", "has_diagnosis")
        self.assertEqual(result['value'], "Diabetes")

    def test_predict(self):
        self.nmn.learn([[1, 2], [3, 4]], [0, 1])  # Mock training
        prediction = self.nmn.predict([[1, 2]])
        self.assertIn(prediction, [0, 1])
