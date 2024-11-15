from graphfusion.nmn_core import NeuralMemoryNetwork

def main():
    # Initialize the NeuralMemoryNetwork
    nmn = NeuralMemoryNetwork()

    # Define some example case data
    case_text_1 = "Patient shows symptoms of fever, cough, and difficulty breathing."
    case_data_1 = {"case_id": 1, "diagnosis": "COVID-19"}

    case_text_2 = "Patient reports mild fever and sore throat, suspected viral infection."
    case_data_2 = {"case_id": 2, "diagnosis": "Common Cold"}

    # Store cases in memory
    print("Storing case 1 in memory...")
    nmn.store_in_memory(case_text_1, data=case_data_1)

    print("Storing case 2 in memory...")
    nmn.store_in_memory(case_text_2, data=case_data_2)

    # Retrieve similar cases
    print("Retrieving similar cases for a new query...")
    query_text = "Patient complains of fever and cough."
    similar_cases = nmn.retrieve_similar(query_text, top_k=5)
    print("Similar cases retrieved:")
    for case in similar_cases:
        print(case)

    # Generate recommendations
    print("Generating recommendations for a new query...")
    recommendations = nmn.get_recommendations(query_text, top_k=3)
    print("Recommendations:")
    for recommendation in recommendations:
        print(recommendation)

if __name__ == "__main__":
    main()
