"""Demo: pregunta "¿Qué es la entropía?" en modo normal y en streaming.

Código de salida: 0 si al menos un proveedor respondió o si no había claves
configuradas; 1 si se intentó algún proveedor y todos fallaron.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import (
    AnthropicClient,
    AsyncLLMManager,
    BaseLLMClient,
    ChatMessage,
    LLMClientError,
    LLMError,
    OpenAIClient,
    Role,
)

KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# Instanciación directa: ambos clientes comparten la interfaz de BaseLLMClient.
CLIENTS: dict[str, type[BaseLLMClient]] = {"openai": OpenAIClient, "anthropic": AnthropicClient}
SYSTEM_PROMPT = "Responde en español de forma breve y clara."
DEFAULT_PROMPT = "¿Qué es la entropía?"


def format_error(error: LLMError) -> str:
    return (
        f"tipo={error.type} status={error.status_code} "
        f"reintentable={error.retryable} mensaje={error.message}"
    )


async def run_provider(
    provider: str, prompt: str, api_key: str | None = None, direct: bool = False
) -> bool:
    """Ejecuta modo normal y streaming. Devuelve True si el modo normal tuvo éxito."""
    messages = [
        ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=Role.USER, content=prompt),
    ]
    try:
        if direct:  # sin manager: el cliente concreto con su ModelConfig
            config = AsyncLLMManager.config_from_env(provider)
            manager = CLIENTS[provider](config, api_key=api_key)
        else:
            manager = AsyncLLMManager.from_env(provider, api_key=api_key)
    except (ValidationError, ValueError) as exc:
        print(f"[{provider}] Configuración inválida: {exc}")
        return False

    async with manager as llm:
        print(f"\n=== {provider} ({llm.config.model}) - modo normal ===")
        resp = await llm.generate(messages)
        if resp.ok:
            print(resp.content)
            print(
                f"-- modelo={resp.model} finish_reason={resp.finish_reason} "
                f"uso={resp.usage} latencia={resp.latency_ms:.0f}ms intentos={resp.attempts}"
            )
        else:
            print(f"ERROR controlado: {format_error(resp.error)} (intentos={resp.attempts})")

        print(f"\n=== {provider} ({llm.config.model}) - streaming ===")
        try:
            async for chunk in llm.stream(messages):
                print(chunk, end="", flush=True)
            print()
        except LLMClientError as exc:
            print(f"\nERROR controlado: {format_error(exc.error)} (intentos={exc.attempts})")
    return resp.ok


async def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente LLM asíncrono: OpenAI / Anthropic")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "all"],
        default=os.environ.get("LLM_PROVIDER") or "all",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--directo",
        action="store_true",
        help="instancia OpenAIClient/AnthropicClient directamente, sin AsyncLLMManager",
    )
    parser.add_argument(
        "--demo-errores",
        action="store_true",
        help="usa una API key inválida (hace una llamada de red real) para mostrar el error controlado",
    )
    args = parser.parse_args()

    if args.provider not in (*KEY_VARS, "all"):
        print(
            f"Proveedor desconocido: {args.provider!r} (revisa LLM_PROVIDER). "
            f"Opciones: {', '.join(KEY_VARS)}, all.",
            file=sys.stderr,
        )
        return 2

    providers = list(KEY_VARS) if args.provider == "all" else [args.provider]
    attempted, succeeded = 0, 0
    for provider in providers:
        fake_key = "clave-invalida-demo" if args.demo_errores else None
        if not fake_key and not os.environ.get(KEY_VARS[provider]):
            print(f"[{provider}] Se omite: no hay {KEY_VARS[provider]} configurada.")
            continue
        attempted += 1
        succeeded += await run_provider(provider, args.prompt, api_key=fake_key, direct=args.directo)
    return 1 if attempted and not succeeded else 0


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(asyncio.run(main()))
