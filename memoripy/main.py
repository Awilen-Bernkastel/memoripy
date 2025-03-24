# main.py

import sys
from memoripy.interaction_data import InteractionData
from memory_manager import MemoryManager
from .memory_storage.json_storage import JSONStorage
from langchain_ollama import ChatOllama

def main(new_prompt):
    # Define chat and embedding models
    chat_model = ChatOllama                     # Choose 'openai' or 'ollama' for chat
    chat_model_name = "qwen2.5:7b"              # Specific chat model name
    embedding_model = ChatOllama                # Choose 'openai' or 'ollama' for embeddings
    embedding_model_name = "mxbai-embed-large"  # Specific embedding model name

    # Choose your storage option
    storage_option = JSONStorage("interaction_history.json")
    # Or use in-memory storage:
    # from in_memory_storage import InMemoryStorage
    # storage_option = InMemoryStorage()

    # Initialize the MemoryManager with the selected models and storage
    memory_manager = MemoryManager(
        # api_key=api_key,
        chat_model=chat_model,
        chat_model_name=chat_model_name,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
        storage=storage_option
    )

    # Load the last 5 interactions from history (for context)
    short_term, _ = memory_manager.load_history()
    last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

    # Retrieve relevant past interactions, excluding the last 5
    relevant_interactions = memory_manager.retrieve_relevant_interactions(new_prompt, exclude_last_n=5)

    # Generate a response using the last interactions and retrieved interactions
    response_chunks = memory_manager.generate_response(new_prompt, last_interactions, relevant_interactions, stream=True)

    response = ""
    # Display the response
    for chunk in response_chunks:
        print(str(chunk.content), end="", flush=True)
        response += str(chunk.content)

    # Extract concepts for the new interaction
    combined_text = f"{new_prompt} {response}"
    concepts = memory_manager.extract_concepts(combined_text)

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
        print("Usage: python3 memoripy.py \"<your prompt here>\"")
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
