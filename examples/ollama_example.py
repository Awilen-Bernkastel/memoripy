# Ollama API example:

import sys
from memoripy import MemoryManager, SQLStorage, JSONStorage, InMemoryStorage
from memoripy.interaction_data import InteractionData
from memoripy.model_interfaces.ollama_models import OllamaChatModel, OllamaEmbeddingModel

def main(prompt):
    # Define chat and embedding models
    chat_model_name = "llama3.1:8b" # Specific chat model name
    embedding_model_name = "mxbai-embed-large"  # Specific embedding model name, this is MixedBreadAI's embedding model.

    # Choose your storage option
    storage_option = JSONStorage()
    # Or use another storage option:
    # storage_option = SQLStorage()
    # storage_option = InMemoryStorage()

    # Initialize the MemoryManager with the selected models and storage
    memory_manager = MemoryManager(OllamaChatModel(chat_model_name), OllamaEmbeddingModel(embedding_model_name), storage=storage_option)

    # New user prompt
    new_prompt = prompt

    print("Getting memories...")
    # Load the last 5 interactions from history (for context)
    short_term, _ = memory_manager.load_history()
    last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

    # Retrieve relevant past interactions, excluding the last 5
    relevant_interactions = memory_manager.retrieve_relevant_interactions(new_prompt, exclude_last_n=5)

    print("Generating response...")
    # Generate a response using the last interactions and retrieved interactions
    chunked_response = memory_manager.generate_response(new_prompt, last_interactions, relevant_interactions)

    # Generate a response using the last interactions and retrieved interactions
    print("Generated response:")
    response = ""
    # Display the response
    for chunk in chunked_response:
        print(chunk.content, end="", flush=True)
        response += str(chunk.content)

    print("\nExtracting concepts...")
    # Extract concepts for the new interaction
    combined_text = f"{new_prompt} {response}"
    concepts = memory_manager.extract_concepts(combined_text)
    print("Concepts: [" + ", ".join(concepts) + "]")

    print("Storing new interaction...")
    # Store this new interaction along with its embedding and concepts
    new_embedding = memory_manager.get_embedding(combined_text)

    interaction = InteractionData(
        prompt=new_prompt,
        output=response,
        embedding=new_embedding,
        concepts=concepts
    )

    memory_manager.add_interaction(interaction)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ollama_example.py prompt...")
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
