# Apache 2.0 license, created by Awilen Bernkastel

import logging
from typing import Iterator
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
import numpy as np
import ollama
from .ollama_modelfile_parser import modelfile
from .model import ChatModel, EmbeddingModel
from ..memory_manager import ConceptExtractionResponse
from langchain_core.runnables.utils import Output

class OllamaChatModel(ChatModel):
    def __init__(self, model_name="llama3.1:8b", temperature=None):
        self.model_name = model_name
        self.modelcard = modelfile().parse(ollama.show(model_name))
        self.llm = ChatOllama(model=model_name, temperature=temperature or float(self.modelcard.oparameters.get("temperature", 0)))
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

        self.template = self.modelcard.otemplate

        self.system_prompt_template = PromptTemplate(
            template= self.template + "\n" + self.modelcard.osystem + "\n"
            "{text}",
            input_variables=["text"]
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
        concepts = None
        while concepts is None:
            try:
                response = chain.invoke({"text": text})
                concepts = response.get("concepts", [])
            except OutputParserException:
                continue
        logging.info(f"Concepts extracted: {concepts}")
        self.llm.temperature = old_temperature
        return concepts

class OllamaEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="mxbai-embed-large"):
        self.model_name = model_name
        self.dimension = self.initialize_embedding_dimension()

    def get_embedding(self, text: str) -> np.ndarray:
        response = ollama.embeddings(model=self.model_name, prompt=text)
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        return np.array(embedding)

    def initialize_embedding_dimension(self) -> int:
        test_text = "Test to determine embedding dimension"
        response = ollama.embeddings(
            model=self.model_name,
            prompt=test_text
        )
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)