from abc import ABC, abstractmethod
import numpy as np
from langchain_core.prompts import PromptTemplate

class EmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        """
        pass

    @abstractmethod
    def initialize_embedding_dimension(self) -> int:
        """
        Determine the dimension of the embeddings.
        """
        pass


class ChatModel(ABC):
    @abstractmethod
    def invoke(self, messages: list) -> str:
        """
        Generate a response from the chat model given a list of messages.
        """
        pass

    @abstractmethod
    def extract_concepts(self, text: str) -> list[str]:
        """
        Extract key concepts from the provided text.
        """
        pass

    @abstractmethod
    def stream(self, messages: list) -> str:
        """
        Generate a response from the chat model given a list of messages in streaming mode.
        """
        pass

    def add_custom_prompt(self, prompt: PromptTemplate):
        if not hasattr(self, "customPrompts"):
            self.customPrompt = []
        self.customPrompt.append(prompt)
