"""LLM Backend implementations for agent orchestrator."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM given a prompt."""
        pass


class DummyBackend(LLMBackend):
    """Dummy backend for testing and development."""
    
    def generate_response(self, prompt: str) -> str:
        # Extract question from prompt for dummy response
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].strip()
            return f"DUMMY: {question} | Response generated based on provided context"
        return f"DUMMY: Generated response for: {prompt[:50]}..."


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        return self._client
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content or "No response generated"
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {e}"


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""
    
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError(
                    "Ollama package not installed. Install with: pip install ollama"
                )
        return self._client
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error generating response: {e}"


class LLMCliBackend(LLMBackend):
    """Backend using the llm CLI tool."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using llm CLI."""
        try:
            import subprocess
            result = subprocess.run(
                ["llm", "-m", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"LLM CLI error: {result.stderr}")
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Request timed out"
        except FileNotFoundError:
            return "Error: llm CLI not found. Install with: pip install llm"
        except Exception as e:
            logger.error(f"LLM CLI error: {e}")
            return f"Error generating response: {e}"


def get_backend(backend_name: str, **kwargs) -> LLMBackend:
    """Factory function to create LLM backend instances."""
    backend_name = backend_name.lower()
    
    if backend_name == "dummy":
        return DummyBackend()
    elif backend_name == "openai":
        return OpenAIBackend(**kwargs)
    elif backend_name == "ollama":
        return OllamaBackend(**kwargs)
    elif backend_name == "llm":
        return LLMCliBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")