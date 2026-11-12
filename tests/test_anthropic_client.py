import asyncio
import time
from types import SimpleNamespace as NS

import anthropic
import pytest

from llm_client import ChatMessage, LLMClientError

U = ChatMessage(role="user", content="hola")
S = ChatMessage(role="system", content="sé breve")


def message(blocks, stop="end_turn"):
    return NS(model="claude-test", content=blocks, stop_reason=stop,
              usage=NS(input_tokens=4, output_tokens=6))


def text(t):
    return NS(type="text", text=t)


class Fake:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def __call__(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def install(client, fake):
    client._client.messages.create = fake
    return fake


async def test_generate_system_y_sin_temperature(make_anthropic):
    c = make_anthropic()
    fake = install(c, Fake(message([text("hola"), text(" mundo")])))
    r = await c.generate([S, ChatMessage(role="system", content="en español"), U])
    assert r.ok and r.content == "hola mundo" and r.finish_reason == "end_turn"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (4, 6)
    kw = fake.calls[0]
    assert kw["system"] == "sé breve\n\nen español"
    assert kw["messages"] == [{"role": "user", "content": "hola"}]
    assert kw["max_tokens"] == 4096 and "temperature" not in kw and "thinking" not in kw


async def test_temperature_nunca_se_envia(make_anthropic, caplog):
    with caplog.at_level("WARNING"):
        c = make_anthropic(temperature=0.5)
    fake = install(c, Fake(message([text("x")])))
    await c.generate([U])
    assert "temperature" not in fake.calls[0]
    assert "se ignora" in caplog.text


async def test_sin_system_lo_omite(make_anthropic):
    c = make_anthropic()
    fake = install(c, Fake(message([text("x")])))
    await c.generate([U])
    assert "system" not in fake.calls[0]


async def test_bloques_thinking_se_omiten(make_anthropic):
    c = make_anthropic()
    install(c, Fake(message([NS(type="thinking", thinking="..."), text("respuesta")])))
    assert (await c.generate([U])).content == "respuesta"


async def test_refusal_no_es_error(make_anthropic):
    c = make_anthropic()
    install(c, Fake(message([], stop="refusal")))
    r = await c.generate([U])
    assert r.ok and r.finish_reason == "refusal" and r.content == ""


async def test_sin_mensaje_user_es_bad_request(make_anthropic):
    r = await make_anthropic().generate([S])
    assert r.error.type == "bad_request"


def test_sin_api_key():
    from llm_client import AnthropicClient, ModelConfig
    with pytest.raises(ValueError):
        AnthropicClient(ModelConfig(provider="anthropic", model="m"))


async def test_auth_no_reintenta(make_anthropic, make_status_error):
    c = make_anthropic()
    fake = install(c, Fake(make_status_error(anthropic, "AuthenticationError", 401)))
    r = await c.generate([U])
    assert r.error.type == "authentication" and r.attempts == 1 and len(fake.calls) == 1


async def test_rate_limit_luego_exito(make_anthropic, make_status_error):
    c = make_anthropic()
    rl = lambda: make_status_error(anthropic, "RateLimitError", 429)
    install(c, Fake(rl(), rl(), message([text("ok")])))
    r = await c.generate([U])
    assert r.ok and r.attempts == 3


async def test_overloaded_529_reintentable(make_anthropic, make_status_error):
    c = make_anthropic(max_retries=1)
    fake = install(c, Fake(*[make_status_error(anthropic, "OverloadedError", 529) for _ in range(2)]))
    r = await c.generate([U])
    assert r.error.type == "server" and r.error.retryable and r.attempts == 2


async def test_conexion_y_timeout(make_anthropic, request_):
    c = make_anthropic(max_retries=1)
    fake = install(c, Fake(*[anthropic.APIConnectionError(request=request_) for _ in range(2)],
                           anthropic.APITimeoutError(request=request_)))
    r = await c.generate([U])
    assert r.error.type == "connection" and r.attempts == 2
    c.config.max_retries = 0
    assert (await c.generate([U])).error.type == "timeout"


# ---- streaming ----
class FakeStream:
    def __init__(self, items, fail_with=None):
        self.items, self.fail_with = items, fail_with
        self.text_stream = self._gen()

    async def _gen(self):
        for i in self.items:
            yield i
        if self.fail_with:
            raise self.fail_with

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class StreamFactory:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def install_stream(client, factory):
    client._client.messages.stream = factory
    return factory


async def test_stream_orden(make_anthropic):
    c = make_anthropic()
    f = install_stream(c, StreamFactory(FakeStream(["a", "b", "c"])))
    assert [t async for t in c.stream([S, U])] == ["a", "b", "c"]
    assert f.calls[0]["system"] == "sé breve"


async def test_stream_reintenta_antes_del_primer_chunk(make_anthropic, request_):
    c = make_anthropic()
    f = install_stream(c, StreamFactory(
        FakeStream([], fail_with=anthropic.APIConnectionError(request=request_)), FakeStream(["x"])))
    assert [t async for t in c.stream([U])] == ["x"] and len(f.calls) == 2


async def test_stream_falla_tras_chunk_sin_reintento(make_anthropic, request_):
    c = make_anthropic()
    f = install_stream(c, StreamFactory(
        FakeStream(["x"], fail_with=anthropic.APIConnectionError(request=request_))))
    got = []
    with pytest.raises(LLMClientError) as ei:
        async for t in c.stream([U]):
            got.append(t)
    assert got == ["x"] and ei.value.error.type == "connection" and len(f.calls) == 1


async def test_sse_error_status_200_overloaded_se_reintenta(make_anthropic, request_):
    import httpx2

    resp = httpx2.Response(200, request=request_)
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    err = anthropic.APIStatusError("Overloaded", response=resp, body=body)
    c = make_anthropic()
    f = install_stream(c, StreamFactory(FakeStream([], fail_with=err), FakeStream(["ok"])))
    assert [t async for t in c.stream([U])] == ["ok"] and len(f.calls) == 2


async def test_sse_error_tras_chunk_se_clasifica(make_anthropic, request_):
    import httpx2

    resp = httpx2.Response(200, request=request_)
    body = {"type": "error", "error": {"type": "authentication_error", "message": "x"}}
    err = anthropic.APIStatusError("x", response=resp, body=body)
    c = make_anthropic()
    install_stream(c, StreamFactory(FakeStream(["a"], fail_with=err)))
    with pytest.raises(LLMClientError) as ei:
        [t async for t in c.stream([U])]
    assert ei.value.error.type == "authentication"


async def test_concurrencia_no_bloquea(make_anthropic):
    c = make_anthropic()

    async def slow(**kw):
        await asyncio.sleep(0.3)
        return message([text("x")])

    install(c, slow)
    t0 = time.perf_counter()
    rs = await asyncio.gather(*(c.generate([U]) for _ in range(3)))
    assert all(r.ok for r in rs) and time.perf_counter() - t0 < 0.55
