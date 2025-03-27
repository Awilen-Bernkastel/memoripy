from memoripy import MemoryManager
from memoripy.interaction import Interaction
from memoripy.memory_storage.dynamo_storage import DynamoStorage
from memoripy.model_interfaces.azure_openai_models import AzureOpenAIEmbeddingModel, AzureOpenAIChatModel


def main():
    # Set here your actual Azure OpenAI API key, endpoint and API version
    azure_api_key = ""
    azure_api_endpoint = ""
    azure_api_version = "2024-10-21"

    # Define chat and embedding models
    chat_model_name = "gpt-4o-mini-2024-07-18"
    embedding_model_name = "text-embedding-3-small"

    # Choose your storage option
    storage_option = DynamoStorage("default")
    # Or use in-memory storage:
    # from memoripy import InMemoryStorage
    # storage_option = InMemoryStorage()

    # Initialize the MemoryManager with the selected models and storage
    memory_manager = MemoryManager(
        AzureOpenAIChatModel(
            azure_api_key, azure_api_version, azure_api_endpoint, chat_model_name
        ),
        AzureOpenAIEmbeddingModel(
            azure_api_key, azure_api_version, azure_api_endpoint, embedding_model_name
        ),
        storage=storage_option,
    )

    # New user prompt
    interaction = Interaction(
        prompt = "My name is David"
    )

    # Load the last 5 interactions from history (for context)
    short_term, _ = memory_manager.load_history()
    last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

    # Retrieve relevant past interactions, excluding the last 5
    relevant_interactions = memory_manager.retrieve_relevant_interactions(
        interaction, exclude_last_n=5
    )

    # Generate a response using the last interactions and retrieved interactions
    interaction.output = memory_manager.generate_response(
        interaction, last_interactions, relevant_interactions
    )

    # Display the response
    print(f"Generated response:\n{interaction.output}")

    # Extract concepts for the new interaction
    combined_text = f"{interaction.prompt} {interaction.output}"
    interaction.concepts = memory_manager.extract_concepts(combined_text)

    # Store this new interaction along with its embedding and concepts
    interaction.embedding = memory_manager.get_embedding(combined_text)

    memory_manager.add_interaction(interaction)

if __name__ == "__main__":
    main()
