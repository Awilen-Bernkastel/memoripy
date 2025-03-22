# Apache 2.0 license, Modified by Awilen Bernkastel

import logging
from typing import Iterator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import numpy as np
from .model import ChatModel, EmbeddingModel
from ..memory_manager import ConceptExtractionResponse
from langchain_core.runnables.utils import Output

logger = logging.getLogger("memoripy")

class AzureOpenAIChatModel(ChatModel):
    def __init__(self, api_key, api_version, azure_endpoint, model_name="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = AzureChatOpenAI(azure_deployment=self.model_name, 
                                   api_key=self.api_key,
                                   azure_endpoint=azure_endpoint,
                                   api_version=api_version)
        self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
        self.concept_extraction_prompt_template = PromptTemplate(
            template=(
                "Extract key concepts from the following text in a concise, context-specific manner. "
                "Include only highly relevant and specific concepts.\n"
                "{format_instructions}\n{text}"
            ),
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, messages: list) -> str:
        chain = None
        if hasattr(self, "customPrompt"):
            chain = self.customPrompt[0]
            if len(self.customPrompt) > 1:
                for prompt in self.customPrompt[1:]:
                    chain |= prompt
            chain |= self.llm
        else:
            chain = self.llm
        response = chain.invoke(messages)
        return str(response.content)

    def stream(self, messages: list) -> Iterator[Output]:
        chain = None
        if hasattr(self, "customPrompt"):
            chain = self.customPrompt[0]
            if len(self.customPrompt) > 1:
                for prompt in self.customPrompt[1:]:
                    chain |= prompt
            chain |= self.llm
        else:
            chain = self.llm
        response = chain.stream(messages)
        return response

    def extract_concepts(self, text: str) -> list[str]:
        chain = self.concept_extraction_prompt_template | self.llm | self.parser
        response = chain.invoke({"text": text})
        concepts = response.get("concepts", [])
        logger.info(f"Concepts extracted: {concepts}")
        return concepts

class AzureOpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key, api_version, azure_endpoint, model_name="text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        self.embeddings_model = AzureOpenAIEmbeddings(model=self.model_name, 
                                                      api_key=self.api_key,
                                                      azure_endpoint=azure_endpoint,
                                                    api_version=api_version)

        if model_name == "text-embedding-3-small":
            self.dimension = 1536
        else:
            raise ValueError("Unsupported OpenAI embedding model name for specified dimension.")

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.embeddings_model.embed_query(text)
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        return np.array(embedding)

    def initialize_embedding_dimension(self) -> int:
        return self.dimension