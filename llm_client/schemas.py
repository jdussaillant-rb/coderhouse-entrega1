"""Modelos Pydantic: mensajes de entrada, configuración del modelo y respuesta."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["openai", "anthropic"]


class Role(StrEnum):
    """Roles válidos de un mensaje de chat."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Mensaje de entrada. Se valida que el rol exista y el contenido no esté vacío."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Role
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """Configuración del modelo y de la robustez del cliente (timeout y reintentos)."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)  # None = no se envía
    max_tokens: int = Field(default=4096, gt=0, le=128_000)
    timeout: float = Field(default=60, gt=0)  # segundos
    max_retries: int = Field(default=2, ge=0, le=10)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(BaseModel):
    """Error estructurado, independiente del proveedor."""

    type: Literal[
        "authentication",
        "permission",
        "rate_limit",
        "timeout",
        "connection",
        "bad_request",
        "not_found",
        "server",
        "unknown",
    ]
    message: str
    provider: Provider
    status_code: int | None = None
    retryable: bool = False
    retry_after: float | None = None  # segundos pedidos por el servidor (Retry-After)


class ModelResponse(BaseModel):
    """Resultado de `generate()`. Si algo falla, `error` viene poblado en vez de lanzar."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = None
    attempts: int = 1
    error: LLMError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
