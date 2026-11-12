"""Excepción controlada que se lanza desde `stream()` cuando falla tras los reintentos."""

from __future__ import annotations

from .schemas import LLMError


class LLMClientError(Exception):
    """Envuelve un `LLMError` estructurado (en lugar de la excepción cruda del SDK)."""

    def __init__(self, error: LLMError, attempts: int = 1) -> None:
        super().__init__(f"[{error.provider}:{error.type}] {error.message}")
        self.error = error
        self.attempts = attempts


class InvalidInputError(ValueError):
    """Entrada inválida detectada por nuestras propias validaciones (no exportada)."""
