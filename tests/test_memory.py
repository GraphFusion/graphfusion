import unittest
from graphfusion import Memory

class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()

    def test_store_and_retrieve(self):
        self.memory.store("patient123", "has_diagnosis", "Diabetes")
        result = self.memory.retrieve("patient123", "has_diagnosis")
        self.assertEqual(result['value'], "Diabetes")

    def test_query(self):
        self.memory.store("patient123", "treated_by", "Insulin")
        relationships = self.memory.query("patient123")
        self.assertIn("treated_by", relationships)
