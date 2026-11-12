"""Clase base abstracta con validación de entrada, reintentos y manejo de errores."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from contextlib import aclosing
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Self

from .errors import InvalidInputError, LLMClientError
from .schemas import ChatMessage, LLMError, ModelConfig, ModelResponse, Provider

logger = logging.getLogger(__name__)

BASE_DELAY = 0.5  # segundos
MAX_DELAY = 8.0
MAX_RETRY_AFTER = 20.0  # tope para el header Retry-After


class BaseLLMClient(ABC):
    """Interfaz común para todos los proveedores.

    Decisión de diseño sobre errores:
    - `generate()` NUNCA lanza por fallos de API/red: devuelve un `ModelResponse`
      con `error` poblado.
    - `stream()` es un generador asíncrono; no puede "devolver" un error, así que
      lanza `LLMClientError` (con `.error` estructurado) en vez de la excepción cruda del SDK.
    - Una lista de mensajes vacía es un error de entrada (`bad_request`, no retryable),
      tratado igual que cualquier otro error según el método.
    """

    provider: Provider

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        # Inyectable para que los tests no esperen de verdad.
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    # ---- hooks que implementa cada proveedor ----
    @abstractmethod
    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Una sola llamada al modelo (sin reintentos). Puede lanzar excepciones del SDK."""

    @abstractmethod
    def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Un solo intento de streaming (generador asíncrono de texto)."""

    @abstractmethod
    def _map_error(self, exc: Exception) -> LLMError:
        """Traduce una excepción del SDK a un `LLMError`."""

    @abstractmethod
    async def aclose(self) -> None:
        """Cierra el cliente HTTP subyacente."""

    # ---- API pública ----
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa. No lanza por errores de API o red."""
        start = time.perf_counter()
        attempts = 0
        try:
            self._validate(messages)
        except InvalidInputError as exc:
            return self._error_response(self._bad_request(str(exc)), attempts=0)

        while True:
            attempts += 1
            try:
                response = await self._generate_once(messages)
            except Exception as exc:  # no captura CancelledError/KeyboardInterrupt
                error = self._safe_map(exc)
                if self._should_retry(error, attempts):
                    await self._backoff(error, attempts)
                    continue
                return self._error_response(error, attempts, start)
            response.attempts = attempts
            response.latency_ms = (time.perf_counter() - start) * 1000
            return response

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Genera texto token a token. Lanza `LLMClientError` si falla tras los reintentos.

        Solo se reintenta si todavía no se emitió ningún fragmento: reintentar después
        duplicaría texto ya mostrado.
        """
        try:
            self._validate(messages)
        except InvalidInputError as exc:
            raise LLMClientError(self._bad_request(str(exc)), attempts=0) from exc

        attempts = 0
        while True:
            attempts += 1
            emitted = False
            try:
                # aclosing cierra el stream del SDK aunque el consumidor haga break.
                async with aclosing(self._stream_once(messages)) as chunks:
                    async for chunk in chunks:
                        emitted = True
                        yield chunk
                return
            except Exception as exc:
                error = self._safe_map(exc)
                if not emitted and self._should_retry(error, attempts):
                    await self._backoff(error, attempts)
                    continue
                raise LLMClientError(error, attempts=attempts) from exc

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ---- helpers compartidos ----
    def _validate(self, messages: Sequence[ChatMessage]) -> None:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise InvalidInputError("Se requiere una lista no vacía de ChatMessage.")
        if not all(isinstance(m, ChatMessage) for m in messages):
            raise InvalidInputError("Todos los elementos deben ser instancias de ChatMessage.")

    def _bad_request(self, message: str) -> LLMError:
        return LLMError(type="bad_request", message=message, provider=self.provider)

    def _safe_map(self, exc: Exception) -> LLMError:
        # Solo nuestras propias validaciones (p. ej. sin mensaje de usuario) son bad_request.
        if isinstance(exc, InvalidInputError):
            return self._bad_request(str(exc))
        return self._map_error(exc)

    def _should_retry(self, error: LLMError, attempts: int) -> bool:
        return error.retryable and attempts <= self.config.max_retries

    async def _backoff(self, error: LLMError, attempts: int) -> None:
        if error.retry_after is not None:
            delay = min(error.retry_after, MAX_RETRY_AFTER)  # el servidor manda
        else:
            delay = min(BASE_DELAY * 2 ** (attempts - 1), MAX_DELAY)
            delay += random.uniform(0, delay / 4)  # jitter
        logger.warning(
            "[%s] %s (intento %d/%d); reintentando en %.2fs",
            self.provider, error.type, attempts, self.config.max_retries + 1, delay,
        )
        await self._sleep(delay)

    def _error_response(
        self, error: LLMError, attempts: int, start: float | None = None
    ) -> ModelResponse:
        latency = (time.perf_counter() - start) * 1000 if start is not None else None
        return ModelResponse(
            provider=self.provider,
            model=self.config.model,
            error=error,
            attempts=attempts,
            latency_ms=latency,
        )


def classify_status(status: int | None) -> tuple[str, bool]:
    """Tipo de error y si es reintentable según el código HTTP (común a ambos SDKs)."""
    if status == 401:
        return "authentication", False
    if status == 403:
        return "permission", False
    if status == 404:
        return "not_found", False
    if status == 408:
        return "timeout", True
    if status == 409:
        return "unknown", True  # conflicto transitorio
    if status == 429:
        return "rate_limit", True
    if status in (400, 422):
        return "bad_request", False
    if status is not None and status >= 500:  # incluye 529 (Anthropic "overloaded")
        return "server", True
    return "unknown", False


_BODY_KINDS: dict[str, tuple[str, bool]] = {
    "insufficient_quota": ("rate_limit", False),  # sin saldo: reintentar no sirve
    "overloaded_error": ("server", True),
    "api_error": ("server", True),
    "server_error": ("server", True),
    "rate_limit_error": ("rate_limit", True),
    "rate_limit_exceeded": ("rate_limit", True),
    "authentication_error": ("authentication", False),
    "invalid_api_key": ("authentication", False),
    "invalid_request_error": ("bad_request", False),
}


def _classify_body(exc: Exception) -> tuple[str, bool] | None:
    """Clasifica por el tipo/código de error del cuerpo (p. ej. un evento SSE `error`)."""
    body = getattr(exc, "body", None)
    names: list[object] = [getattr(exc, "code", None), getattr(exc, "type", None)]
    if isinstance(body, dict):
        inner = body.get("error")
        for source in (body, inner if isinstance(inner, dict) else {}):
            names += [source.get("type"), source.get("code")]
    for name in names:
        if isinstance(name, str) and name.lower() in _BODY_KINDS:
            return _BODY_KINDS[name.lower()]
    return None


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    raw = response.headers.get("retry-after") if response is not None else None
    try:
        value = float(raw) if raw is not None else None
    except ValueError:  # formato fecha HTTP: se ignora
        return None
    return value if value is not None and value >= 0 else None


def map_sdk_error(exc: Exception, provider: Provider, sdk: object) -> LLMError:
    """Mapeo común: `sdk` es el módulo (`openai` o `anthropic`), ambos con la misma jerarquía.

    Se evalúa de lo más específico a lo más general; `APITimeoutError` hereda de
    `APIConnectionError`, por eso va antes.
    """
    message = _safe_message(exc)
    if isinstance(exc, sdk.APITimeoutError):  # type: ignore[attr-defined]
        return LLMError(type="timeout", message=message, provider=provider, retryable=True)
    if isinstance(exc, sdk.APIConnectionError):  # type: ignore[attr-defined]
        return LLMError(type="connection", message=message, provider=provider, retryable=True)
    if isinstance(exc, sdk.APIError):  # type: ignore[attr-defined]
        status = getattr(exc, "status_code", None)
        body = _classify_body(exc)
        if status is not None and status >= 400:
            kind, retryable = classify_status(status)
            if kind == "rate_limit" and body == ("rate_limit", False):
                retryable = False
        else:
            # Error a mitad de stream: Anthropic lo entrega como APIStatusError con
            # status 200 y OpenAI como APIError sin status; el tipo va solo en el cuerpo.
            kind, retryable = body or ("unknown", False)
        return LLMError(
            type=kind, message=message, provider=provider, status_code=status,
            retryable=retryable, retry_after=_retry_after(exc) if retryable else None,
        )
    return LLMError(type="unknown", message=message, provider=provider, retryable=False)


_KEY_PATTERN = re.compile(r"sk-[\w\-*]+")


def _safe_message(exc: Exception) -> str:
    """Mensaje del SDK truncado y sin claves: algunos SDK devuelven la clave enmascarada
    en el 401 (p. ej. `sk-proj-****abcd`), así que se redacta todo lo que parezca una."""
    return _KEY_PATTERN.sub("sk-***", str(exc) or exc.__class__.__name__)[:500]
