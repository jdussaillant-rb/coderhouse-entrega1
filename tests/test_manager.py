import pytest
from pydantic import ValidationError

from llm_client import AnthropicClient, AsyncLLMManager, LLMClientError, ModelConfig, OpenAIClient


def test_por_defecto_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    m = AsyncLLMManager.from_env()
    assert isinstance(m.client, OpenAIClient) and m.config.model == "gpt-5-mini"


def test_provider_y_modelo_desde_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    monkeypatch.setenv("LLM_MAX_TOKENS", "256")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    m = AsyncLLMManager.from_env()
    assert isinstance(m.client, AnthropicClient)
    assert (m.config.model, m.config.max_tokens, m.config.max_retries) == ("claude-x", 256, 5)


def test_argumento_gana_sobre_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(AsyncLLMManager.from_env("openai").client, OpenAIClient)


def test_temperatura_invalida(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("LLM_TEMPERATURE", "3")
    with pytest.raises(ValidationError):
        AsyncLLMManager.from_env()


def test_sin_clave_falla_claro():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AsyncLLMManager.from_env("openai")


async def test_delegacion_y_prompt_str():
    from types import SimpleNamespace as NS

    async with AsyncLLMManager(ModelConfig(provider="openai", model="m"), api_key="k") as m:
        seen = {}

        async def fake(**kw):
            seen.update(kw)
            return NS(model="m", choices=[NS(message=NS(content="r"), finish_reason="stop")], usage=None)

        m.client._client.chat.completions.create = fake
        r = await m.generate("hola")
    assert r.ok and r.content == "r" and seen["messages"] == [{"role": "user", "content": "hola"}]


async def test_prompt_vacio_generate_devuelve_bad_request():
    async with AsyncLLMManager(ModelConfig(provider="openai", model="m"), api_key="k") as m:
        r = await m.generate("   ")
    assert not r.ok and r.error.type == "bad_request"


async def test_prompt_vacio_stream_lanza_llmclienterror():
    async with AsyncLLMManager(ModelConfig(provider="openai", model="m"), api_key="k") as m:
        with pytest.raises(LLMClientError) as ei:
            [t async for t in m.stream("")]
    assert ei.value.error.type == "bad_request"


def test_proveedor_desconocido_en_env():
    with pytest.raises(ValueError, match="gemini"):
        AsyncLLMManager.from_env("gemini")


def test_from_env_acepta_api_key_explicita():
    m = AsyncLLMManager.from_env("anthropic", api_key="clave-demo")
    assert isinstance(m.client, AnthropicClient)
