# Apache 2.0 license, Modified by Awilen Bernkastel

from datetime import datetime
import numpy as np
import time
import uuid
from pydantic import BaseModel, Field
from .memory_storage.in_memory_storage import InMemoryStorage
from langchain_core.messages import HumanMessage, SystemMessage
from .memory_store import MemoryStore
from .model_interfaces.model import ChatModel, EmbeddingModel
import logging
from .interaction import Interaction

class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")

logger = logging.getLogger("memoripy")

class MemoryManager:
    """
    Manages the memory store, including loading and saving history,
    adding interactions, retrieving relevant interactions, and generating responses.
    """

    def __init__(self, chat_model: ChatModel, embedding_model: EmbeddingModel, storage=None, prompt_elements=None):
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.prompt_elements = {
            "intro": "",
            "intro_stm": "This is the current conversation: ",
            "stm_user": "Previous prompt: ",
            "stm_agent": "Previous output: ",
            "outro_stm": "",
            "intro_ltm": "Those are past interactions that you can use as additional context: ",
            "ltm_user": "Relevant prompt: ",
            "ltm_agent": "Relevant output: ",
            "outro_ltm": "",
            "outro": "",
            "no_history": "No previous interactions available.",
            "system": "You're a helpful assistant.",
            "prompt_intro": "Current prompt: ",
            "response_intro": "You: "
        }
        if prompt_elements is not None:
            self.prompt_elements.update(prompt_elements)

        # Initialize memory store with the correct dimension
        self.dimension = self.embedding_model.initialize_embedding_dimension()
        try:
            self.memory_store = MemoryStore(dimension=self.dimension) # FAISS-enabled
        except TypeError:
            self.memory_store = MemoryStore() # Non-FAISS

        if storage is None:
            self.storage = InMemoryStorage()
        else:
            self.storage = storage

        self.initialize_memory()

    def standardize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Standardize embedding to the target dimension by padding with zeros or truncating.
        """
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            # Pad with zeros
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
            # Truncate to match target dimension
            return embedding[:self.dimension]

    def load_history(self):
        return self.storage.load_history()

    def save_memory_to_history(self):
        self.storage.save_memory_to_history(self.memory_store)

    # def add_interaction(self, prompt: str, output: str, embedding: np.ndarray, concepts: list[str]):
    def add_interaction(self, interaction: Interaction):
        timestamp = time.time()
        interaction.id = str(uuid.uuid4())
        interaction.timestamp = timestamp
        interaction.last_accessed = timestamp
        interaction.concepts = [str(concept) for concept in interaction.concepts]
        interaction.access_count = 1
        interaction.decay_factor = 1.0
        self.memory_store.add_interaction(interaction)
        self.save_memory_to_history()

    def get_embedding(self, text: str) -> np.ndarray:
        logger.info(f"Generating embedding for the provided text...")
        embedding = self.embedding_model.get_embedding(text)
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        standardized_embedding = self.standardize_embedding(embedding)
        return np.array(standardized_embedding).reshape(1, -1)

    def extract_concepts(self, text: str) -> list[str]:
        logger.info("Extracting key concepts from the provided text...")
        return self.chat_model.extract_concepts(text)

    def initialize_memory(self):
        short_term, long_term = self.load_history()
        for interaction in short_term:
            # Standardize the dimension of each interaction's embedding
            interaction.embedding = self.standardize_embedding(np.array(interaction['embedding']))
            self.memory_store.add_interaction(interaction)
        self.memory_store.long_term_memory.extend(long_term)

        self.memory_store.cluster_interactions()
        logger.info(f"Memory initialized with {len(self.memory_store.short_term_memory)} interactions in short-term and {len(self.memory_store.long_term_memory)} in long-term.")

    def retrieve_relevant_interactions(self, interaction: Interaction, similarity_threshold=0.4, exclude_last_n=0) -> list:
        interaction.embedding = self.get_embedding(interaction.prompt)
        interaction.concepts = self.extract_concepts(interaction.prompt)
        return self.memory_store.retrieve(interaction, similarity_threshold, exclude_last_n=exclude_last_n)

    def generate_response(self, query_interaction: Interaction, last_interactions: list, retrievals: list, context_window=3, stream=True) -> str:
        context = self.prompt_elements["intro"]
        if retrievals:
            context += self.prompt_elements["intro_ltm"]
            retrieved_context_interactions = retrievals[:context_window]
            retrieved_context = "\n".join([f"({datetime.fromtimestamp(r["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')}) {self.prompt_elements["ltm_user"]}\
                                           {r['prompt']}\n{self.prompt_elements["ltm_agent"]}{r['output']}" for r in retrieved_context_interactions])
            logger.info(f"Using the following retrieved interactions as context for response generation:\n{retrieved_context}")
            context += "\n" + retrieved_context
            context += self.prompt_elements["outro_ltm"]

        if last_interactions:
            context += self.prompt_elements["intro_stm"]
            context_interactions = last_interactions[-context_window:]
            context += "\n".join([f"({datetime.fromtimestamp(r["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')}) {self.prompt_elements["stm_user"]}{r['prompt']}\n{self.prompt_elements["stm_agent"]}{r['output']}" for r in context_interactions])
            logger.info(f"Using the following last interactions as context for response generation:\n{context}")
            context += self.prompt_elements["outro_stm"]
        elif context == self.prompt_elements["intro"]:
            context = self.prompt_elements["no_history"]
            logger.info(context)

        if self.prompt_elements["outro"] != "":
            context += "\n" + self.prompt_elements["outro"]

        messages = [
            SystemMessage(content=self.prompt_elements["system"]),
            HumanMessage(content=f"{context}\n({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}){self.prompt_elements["prompt_intro"]}{query_interaction.prompt} {self.prompt_elements["response_intro"]}")
        ]

        if stream:
            response = self.chat_model.stream(messages)
        else:
            response = self.chat_model.invoke(messages)

        return response
