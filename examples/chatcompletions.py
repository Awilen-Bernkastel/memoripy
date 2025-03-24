# General chat completions endpoint example (using OpenRouter):

from memoripy import MemoryManager, JSONStorage
from memoripy.interaction_data import InteractionData
from memoripy.model_interfaces.chat_completion_models import ChatCompletionsModel
from memoripy.model_interfaces.ollama_models import OllamaEmbeddingModel

def main():
    # Set API endpoint and key
    api_endpoint = "https://openrouter.ai/api/v1"
    # Replace 'your-api-key' with your actual API key for the chat completions service you use
    api_key = "your-api-key"
    if not api_key or api_key == "your-api-key":
        raise ValueError("Please set your API key.")

    # Define chat and embedding models
    chat_model_name = "nousresearch/hermes-3-llama-3.1-405b:free" # Specific chat model name
    embedding_model_name = "mxbai-embed-large"  # Specific embedding model name

    # Choose your storage option
    storage_option = JSONStorage("interaction_history.json")
    # Or use in-memory storage:
    # from memoripy import InMemoryStorage
    # storage_option = InMemoryStorage()

    # Initialize the MemoryManager with the selected models and storage
    memory_manager = MemoryManager(ChatCompletionsModel(api_endpoint, api_key, chat_model_name), OllamaEmbeddingModel(embedding_model_name), storage=storage_option)

    # New user prompt
    new_prompt = "My name is Khazar"

    # Load the last 5 interactions from history (for context)
    short_term, _ = memory_manager.load_history()
    last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

    # Retrieve relevant past interactions, excluding the last 5
    relevant_interactions = memory_manager.retrieve_relevant_interactions(new_prompt, exclude_last_n=5)

    # Generate a response using the last interactions and retrieved interactions
    response = memory_manager.generate_response(new_prompt, last_interactions, relevant_interactions)

    # Display the response
    print(f"Generated response:\n{response}")

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
    main()
