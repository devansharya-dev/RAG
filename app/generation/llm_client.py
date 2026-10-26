
"""Ollama chat client with retry and timeout handling."""

import time
from typing import Any

import httpx
import ollama

from app.core.config import get_settings


class LLMClient:
    """Generate responses using a locally running Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 120.0,
        retries: int = 2,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.OLLAMA_CHAT_MODEL
        self.timeout = timeout
        self.retries = retries
        self.client = ollama.Client(
            host=settings.OLLAMA_BASE_URL,
            timeout=timeout,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Generate a chat response, retrying transient failures."""
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )
                return response.message.content.strip()

            except (ollama.ResponseError, httpx.HTTPError, OSError) as exc:
                last_error = exc

                # Model-not-found errors are configuration errors, not transient.
                if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
                    raise RuntimeError(
                        f"Ollama model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}"
                    ) from exc

                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Could not generate an answer using Ollama: {last_error}"
        ) from last_error
