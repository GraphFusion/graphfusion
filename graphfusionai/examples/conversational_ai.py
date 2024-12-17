from graphfusionai.graph import Graph

# Initialize the conversation graph
conversation_graph = Graph()

# Function to add knowledge to the graph during conversation
def add_conversation_knowledge(user_input, intent, response):
    conversation_graph.add_knowledge(user_input, 'intent', intent)
    conversation_graph.add_knowledge(user_input, 'response', response)

# Function to simulate conversation
def respond_to_user(user_input):
    # Check for any related intents in the graph (for example, greeting)
    related_intents = conversation_graph.query_graph(source=user_input, edge_type='intent')

    if related_intents:
        # If there's a match, respond accordingly
        for intent in related_intents:
            if intent[1] == 'greeting':
                return "Hello! How can I assist you today?"
            elif intent[1] == 'question':
                return "That sounds like an interesting question. Let me find more information for you."
            else:
                return "I'm not sure about that. Can you clarify?"
    else:
        # If no related intents are found, ask the user to clarify
        return "I didn't quite understand that. Could you rephrase?"

# Add some knowledge to the conversation graph
add_conversation_knowledge('Hi', 'greeting', 'Hello! How can I assist you today?')
add_conversation_knowledge('What is AI?', 'question', 'AI is the simulation of human intelligence in machines.')

# Start the conversation
print("Conversational AI: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Conversational AI: Goodbye!")
        break

    response = respond_to_user(user_input)
    print(f"Conversational AI: {response}")
