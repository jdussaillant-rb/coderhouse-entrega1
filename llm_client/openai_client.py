"""Cliente asíncrono para OpenAI (Chat Completions)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI

from .base import BaseLLMClient, map_sdk_error
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, TokenUsage

DEFAULT_MODEL = "gpt-5-mini"


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Falta la API key de OpenAI (variable OPENAI_API_KEY).")
        # max_retries=0: los reintentos los hacemos nosotros, para no duplicarlos.
        self._client = AsyncOpenAI(api_key=key, timeout=config.timeout, max_retries=0)

    def _params(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "max_completion_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        return params

    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        completion = await self._client.chat.completions.create(**self._params(messages))
        choice = completion.choices[0] if completion.choices else None
        usage = None
        if completion.usage is not None:
            usage = TokenUsage(
                input_tokens=completion.usage.prompt_tokens or 0,
                output_tokens=completion.usage.completion_tokens or 0,
            )
        return ModelResponse(
            provider=self.provider,
            model=getattr(completion, "model", None) or self.config.model,
            content=(choice.message.content or "") if choice else "",
            finish_reason=choice.finish_reason if choice else None,
            usage=usage,
        )

    async def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._params(messages), stream=True, stream_options={"include_usage": True}
        )
        async with stream:  # cierra la conexión aunque el consumidor corte el stream
            async for chunk in stream:
                # El último chunk (uso de tokens) llega con `choices` vacío.
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    def _map_error(self, exc: Exception) -> LLMError:
        return map_sdk_error(exc, self.provider, openai)

    async def aclose(self) -> None:
        await self._client.close()
