import pytest
from pydantic import ValidationError

from llm_client import ChatMessage, ModelConfig, ModelResponse


def cfg(**kw):
    return ModelConfig(provider="openai", model="m", **kw)


@pytest.mark.parametrize("t", [0, 2, 1.0, None])
def test_temperatura_valida(t):
    assert cfg(temperature=t).temperature == t


@pytest.mark.parametrize("kw", [{"temperature": 2.5}, {"temperature": -0.1}, {"max_tokens": 0},
                                {"timeout": 0}, {"max_retries": 11}, {"foo": 1}])
def test_config_invalida(kw):
    with pytest.raises(ValidationError):
        cfg(**kw)


def test_proveedor_desconocido():
    with pytest.raises(ValidationError):
        ModelConfig(provider="gemini", model="m")


def test_mensaje_valido_y_strip():
    assert ChatMessage(role="user", content="  hola ").content == "hola"


@pytest.mark.parametrize("kw", [{"role": "user", "content": "   "}, {"role": "robot", "content": "x"},
                                {"role": "user", "content": "x", "extra": 1}])
def test_mensaje_invalido(kw):
    with pytest.raises(ValidationError):
        ChatMessage(**kw)


def test_response_ok():
    assert ModelResponse(provider="openai", model="m").ok
