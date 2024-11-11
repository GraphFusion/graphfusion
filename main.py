# main.py
from graphfusion.memory import Memory
from graphfusion.learning import Learning
from graphfusion.nmn import NeuralMemoryNetwork
from graphfusion.utils import get_version 

def main():
    # Initialize the NMN
    nmn = NeuralMemoryNetwork()

    # Store some data in memory
    nmn.store_data("entity1", "relation", "value")

    # Retrieve and print stored data
    print("Retrieve data:", nmn.retrieve_data("entity1", "relation"))

    # Simulate training data
    features = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    labels = [0, 1, 1]
    nmn.learn(features, labels)

    # Make a prediction
    prediction = nmn.predict([[0.1, 0.2]])
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()
