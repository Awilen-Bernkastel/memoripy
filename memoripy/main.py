# main.py

import sys
from memoripy.interaction import Interaction
from memory_manager import MemoryManager
from .memory_storage.json_storage import JSONStorage
from langchain_ollama import ChatOllama

def main(new_prompt):
    # Define chat and embedding models
    chat_model = ChatOllama                     # Choose 'openai' or 'ollama' for chat
    chat_model_name = "llama3.1:8b"             # Specific chat model name
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

    interaction = Interaction(
        prompt=new_prompt,
        output=""
    )

    # Load the last 5 interactions from history (for context)
    short_term, _ = memory_manager.load_history()
    last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

    # Retrieve relevant past interactions, excluding the last 5
    relevant_interactions = memory_manager.retrieve_relevant_interactions(interaction, exclude_last_n=5)

    # Generate a response using the last interactions and retrieved interactions
    response_chunks = memory_manager.generate_response(interaction, last_interactions, relevant_interactions, stream=True)

    # Display the response
    for chunk in response_chunks:
        print(str(chunk.content), end="", flush=True)
        interaction.output += str(chunk.content)

    # Extract concepts for the new interaction
    combined_text = f"{interaction.prompt} {interaction.output}"
    interaction.concepts = memory_manager.extract_concepts(combined_text)

    # Store this new interaction along with its embedding and concepts
    interaction.embedding = memory_manager.get_embedding(combined_text)

    memory_manager.add_interaction(interaction)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 memoripy.py \"<your prompt here>\"")
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
