"""Cliente asíncrono para Anthropic (Messages API)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from .errors import InvalidInputError
from .base import BaseLLMClient, map_sdk_error
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Role, TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Falta la API key de Anthropic (variable ANTHROPIC_API_KEY).")
        self._client = AsyncAnthropic(api_key=key, timeout=config.timeout, max_retries=0)
        if config.temperature is not None:
            logger.warning(
                "temperature=%s se ignora con Anthropic: los modelos actuales no aceptan "
                "parámetros de muestreo.", config.temperature,
            )

    def _params(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        """La API de Anthropic recibe el `system` aparte del resto de mensajes."""
        system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        chat = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        if not any(m["role"] == "user" for m in chat):
            raise InvalidInputError("Se requiere al menos un mensaje con rol 'user'.")
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": chat,
        }
        if system:
            params["system"] = system
        return params

    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        message = await self._client.messages.create(**self._params(messages))
        # Se omiten los bloques "thinking"; solo se concatena el texto.
        text = "".join(b.text for b in message.content if b.type == "text")
        usage = None
        if message.usage is not None:
            usage = TokenUsage(
                input_tokens=message.usage.input_tokens or 0,
                output_tokens=message.usage.output_tokens or 0,
            )
        return ModelResponse(
            provider=self.provider,
            model=getattr(message, "model", None) or self.config.model,
            content=text,
            finish_reason=message.stop_reason,  # incluye "refusal": no es un error
            usage=usage,
        )

    async def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._params(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    def _map_error(self, exc: Exception) -> LLMError:
        return map_sdk_error(exc, self.provider, anthropic)

    async def aclose(self) -> None:
        await self._client.close()
