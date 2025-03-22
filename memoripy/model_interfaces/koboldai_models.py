# Apache 2.0 license, Modified by Awilen Bernkastel

from typing import Iterator
import numpy as np
import logging
from langchain_community.llms.koboldai import KoboldApiLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from .model import ChatModel, EmbeddingModel
from ..memory_manager import ConceptExtractionResponse
from langchain_core.runnables.utils import Output

logger = logging.getLogger("memoripy")

class KoboldAiChatModel(ChatModel):
    def __init__(self, api_key=None, model_name="llama3.1:8b", endpoint="http://localhost:5000"):
        self.api_key = api_key
        self.model_name = model_name
        # Initialize the KoboldApiLLM with your API key and model name
        self.llm = KoboldApiLLM(endpoint=endpoint)
        self.llm = KoboldApiLLM(api_key=self.api_key, model_name=self.model_name)
        self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
        self.concept_extraction_prompt_template = PromptTemplate(
            template=(
                "Please analyze the following text and provide a list of key concepts that are unique to this content. "
                "Return only the core concepts that best capture the text's meaning.\n"
                "{format_instructions}\n{text}"
            ),
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, messages: list) -> str:
        chain = self.system_prompt_template
        if hasattr(self, "customPrompt"):
            for prompt in self.customPrompt:
                chain |= prompt
        chain |= self.llm
        response = chain.invoke(messages)
        return str(response.content)

    def stream(self, messages: list) -> Iterator[Output]:
        chain = self.system_prompt_template
        if hasattr(self, "customPrompt"):
            for prompt in self.customPrompt:
                chain |= prompt
        chain |= self.llm
        response = chain.stream(messages)
        return response

    def extract_concepts(self, text: str) -> list[str]:
        old_temperature = self.llm.temperature
        self.llm.temperature = 0
        chain = self.concept_extraction_prompt_template | self.llm | self.parser
        response = chain.invoke({"text": text})
        concepts = response.get("concepts", [])
        logger.info(f"Concepts extracted: {concepts}")
        self.llm.temperature = old_temperature
        return concepts

class KoboldEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key, model_name="text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name
        self.embeddings_model = KoboldApiLLM(model=model_name, api_key=self.api_key)
