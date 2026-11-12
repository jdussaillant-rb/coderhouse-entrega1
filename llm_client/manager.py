"""`AsyncLLMManager`: elige el proveedor según la configuración."""

from __future__ import annotations

import os
from contextlib import aclosing
from collections.abc import AsyncIterator, Sequence
from typing import Self

from . import anthropic_client, openai_client
from .anthropic_client import AnthropicClient
from pydantic import ValidationError

from .base import BaseLLMClient
from .errors import InvalidInputError, LLMClientError
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Role

PROVIDERS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}
DEFAULT_MODELS = {
    "openai": openai_client.DEFAULT_MODEL,
    "anthropic": anthropic_client.DEFAULT_MODEL,
}
Prompt = str | Sequence[ChatMessage]


class AsyncLLMManager:
    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        try:
            client_cls = PROVIDERS[config.provider]
        except KeyError:
            raise ValueError(
                f"Proveedor desconocido: {config.provider!r}. Disponibles: {sorted(PROVIDERS)}"
            ) from None
        self.config = config
        self.client: BaseLLMClient = client_cls(config, api_key=api_key)  # type: ignore[call-arg]

    @staticmethod
    def config_from_env(provider: str | None = None) -> ModelConfig:
        """Lee la configuración de variables de entorno (ValidationError si son inválidas)."""
        provider = provider or os.environ.get("LLM_PROVIDER") or "openai"
        if provider not in PROVIDERS:
            raise ValueError(
                f"Proveedor desconocido: {provider!r}. Disponibles: {sorted(PROVIDERS)}"
            )
        values: dict[str, object] = {
            "provider": provider,
            "model": os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider],
        }
        for field, var in (
            ("temperature", "LLM_TEMPERATURE"),
            ("max_tokens", "LLM_MAX_TOKENS"),
            ("timeout", "LLM_TIMEOUT"),
            ("max_retries", "LLM_MAX_RETRIES"),
        ):
            if os.environ.get(var):  # vacío = no definido
                values[field] = os.environ[var]
        # Pydantic (modo lax) convierte los strings a número y valida los rangos.
        return ModelConfig.model_validate(values)

    @classmethod
    def from_env(cls, provider: str | None = None, api_key: str | None = None) -> Self:
        return cls(cls.config_from_env(provider), api_key=api_key)

    @staticmethod
    def _to_messages(prompt: Prompt) -> Sequence[ChatMessage]:
        if isinstance(prompt, str):
            try:
                return [ChatMessage(role=Role.USER, content=prompt)]
            except ValidationError as exc:
                raise InvalidInputError("El prompt no puede estar vacío.") from exc
        return prompt

    async def generate(self, prompt: Prompt) -> ModelResponse:
        """Nunca lanza por entrada inválida: devuelve un `bad_request` en `error`."""
        try:
            messages = self._to_messages(prompt)
        except InvalidInputError as exc:
            return self.client._error_response(self.client._bad_request(str(exc)), attempts=0)
        return await self.client.generate(messages)

    async def stream(self, prompt: Prompt) -> AsyncIterator[str]:
        """Lanza `LLMClientError` (también para entrada inválida)."""
        try:
            messages = self._to_messages(prompt)
        except InvalidInputError as exc:
            raise LLMClientError(self.client._bad_request(str(exc)), attempts=0) from exc
        async with aclosing(self.client.stream(messages)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
