# Apache 2.0 license, Modified by Awilen Bernkastel

import logging
from typing import Iterator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .model import ChatModel
from ..memory_manager import ConceptExtractionResponse
from langchain_core.runnables.utils import Output

class ChatCompletionsModel(ChatModel):
    def __init__(self, api_endpoint: str, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatOpenAI(openai_api_base = api_endpoint, openai_api_key = api_key, model_name = model_name)
        self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
        self.concept_extraction_prompt_template = PromptTemplate(
            template=(
                "Extract key concepts from the following text in a concise, context-specific manner. "
                "Include only the most highly relevant and specific core concepts that best capture the text's meaning. "
                "Return nothing but the JSON string.\n"
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
        logging.info(f"Concepts extracted: {concepts}")
        return concepts

class OpenRouterChatModel(ChatCompletionsModel):
    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_endpoint="https://openrouter.ai/api/v1", api_key=api_key, model_name=model_name)
