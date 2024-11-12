import torch
from transformers import AutoModel, AutoTokenizer

class NeuralMemoryNetwork:
    def __init__(self, model_name="bert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def compute_similarity(self, embedding1, embedding2):
        return torch.cosine_similarity(embedding1, embedding2)

    def analyze_input(self, input_text):
        return {"embedding": self.embed_text(input_text), "text": input_text}
