import httpx2
import pytest

from llm_client import AnthropicClient, ModelConfig, OpenAIClient


@pytest.fixture(autouse=True)
def _env_limpio(monkeypatch):
    """Los tests no dependen del entorno real: se limpian claves y variables LLM_*."""
    for var in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "OPENAI_MODEL",
        "ANTHROPIC_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_TIMEOUT",
        "LLM_MAX_RETRIES",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def request_():
    return httpx2.Request("POST", "https://example.test/v1")


@pytest.fixture
def make_status_error(request_):
    def make(sdk, cls_name, status, body=None):
        response = httpx2.Response(status, request=request_)
        return getattr(sdk, cls_name)("error simulado", response=response, body=body)
    return make


async def _no_sleep(_):
    return None


def _build(cls, provider, model, **cfg):
    client = cls(ModelConfig(provider=provider, model=model, **cfg), api_key="clave-de-prueba")
    client._sleep = _no_sleep  # backoff instantáneo
    return client


@pytest.fixture
def make_openai():
    return lambda **cfg: _build(OpenAIClient, "openai", "gpt-test", **cfg)


@pytest.fixture
def make_anthropic():
    return lambda **cfg: _build(AnthropicClient, "anthropic", "claude-test", **cfg)
