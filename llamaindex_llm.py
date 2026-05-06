from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str) -> str:
        ...

    def stream_complete(self, prompt: str):
        ...


@dataclass
class DeterministicLLM:
    name: str = "deterministic-local-simulator"

    def complete(self, prompt: str) -> str:
        memory_lines = [line for line in prompt.splitlines() if line.startswith("- [")]
        return " ".join(memory_lines)

    def stream_complete(self, prompt: str):
        text = self.complete(prompt)
        if text:
            yield text


class LlamaIndexOllamaLLM:
    """
    Open-source LLM adapter using LlamaIndex + Ollama.

    Example setup:
      ollama pull llama3.2:1b
      pip install llama-index llama-index-llms-ollama

    The import is lazy so tests can run without optional dependencies.
    """

    def __init__(self, model: str = "llama3.2:1b", request_timeout: float = 120.0) -> None:
        from llama_index.core import Settings
        from llama_index.llms.ollama import Ollama

        self.name = f"llamaindex-ollama:{model}"
        self.llm = Ollama(model=model, request_timeout=request_timeout)
        Settings.llm = self.llm

    def complete(self, prompt: str) -> str:
        response = self.llm.complete(prompt)
        return str(response)

    def stream_complete(self, prompt: str):
        if hasattr(self.llm, "stream_complete"):
            for chunk in self.llm.stream_complete(prompt):
                yield str(getattr(chunk, "delta", chunk))
            return
        yield self.complete(prompt)


class LlamaIndexHuggingFaceEmbeddingConfig:
    """
    Local embedding config for LlamaIndex vector stores.

    Example setup:
      pip install llama-index llama-index-embeddings-huggingface
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.name = f"llamaindex-huggingface-embedding:{model_name}"
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)
        Settings.embed_model = self.embed_model

    def get_text_embedding(self, text: str):
        return self.embed_model.get_text_embedding(text)

    def get_query_embedding(self, query: str):
        return self.embed_model.get_query_embedding(query)
