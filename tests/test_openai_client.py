import asyncio
import random
import time
from types import SimpleNamespace as NS

import httpx2
import openai
import pytest

from llm_client import BaseLLMClient, ChatMessage, LLMClientError

MSGS = [ChatMessage(role="user", content="hola")]


def completion(text="ok"):
    return NS(
        model="gpt-test",
        choices=[NS(message=NS(content=text), finish_reason="stop")],
        usage=NS(prompt_tokens=3, completion_tokens=5),
    )


class Fake:
    """Reemplaza `chat.completions.create`: devuelve/lanza según la lista dada."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def __call__(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def install(client, fake):
    client._client.chat.completions.create = fake
    return fake


async def test_generate_ok(make_openai):
    c = make_openai()
    fake = install(c, Fake(completion("hola!")))
    r = await c.generate(MSGS)
    assert r.ok and r.content == "hola!" and r.finish_reason == "stop"
    assert (r.usage.input_tokens, r.usage.output_tokens) == (3, 5)
    assert r.attempts == 1 and r.latency_ms is not None
    assert fake.calls[0]["max_completion_tokens"] == 4096
    assert "max_tokens" not in fake.calls[0] and "temperature" not in fake.calls[0]


async def test_temperature_enviada(make_openai):
    c = make_openai(temperature=0.3)
    fake = install(c, Fake(completion()))
    await c.generate(MSGS)
    assert fake.calls[0]["temperature"] == 0.3


async def test_lista_vacia_es_bad_request(make_openai):
    r = await make_openai().generate([])
    assert r.error.type == "bad_request" and not r.error.retryable


def test_sin_api_key():
    from llm_client import OpenAIClient, ModelConfig
    with pytest.raises(ValueError):
        OpenAIClient(ModelConfig(provider="openai", model="m"))


async def test_auth_no_reintenta(make_openai, make_status_error):
    c = make_openai()
    fake = install(c, Fake(make_status_error(openai, "AuthenticationError", 401)))
    r = await c.generate(MSGS)
    assert r.error.type == "authentication" and r.error.status_code == 401
    assert r.attempts == 1 and len(fake.calls) == 1


async def test_rate_limit_dos_veces_y_exito(make_openai, make_status_error):
    c = make_openai()
    rl = lambda: make_status_error(openai, "RateLimitError", 429)
    install(c, Fake(rl(), rl(), completion()))
    r = await c.generate(MSGS)
    assert r.ok and r.attempts == 3


async def test_insuficiente_cuota_no_reintenta(make_openai, make_status_error):
    c = make_openai()
    err = make_status_error(openai, "RateLimitError", 429, body={"code": "insufficient_quota", "message": "x"})
    fake = install(c, Fake(err))
    r = await c.generate(MSGS)
    assert r.error.type == "rate_limit" and not r.error.retryable and len(fake.calls) == 1


async def test_conexion_persistente(make_openai, request_):
    c = make_openai(max_retries=2)
    fake = install(c, Fake(*[openai.APIConnectionError(request=request_) for _ in range(3)]))
    r = await c.generate(MSGS)
    assert r.error.type == "connection" and r.attempts == 3 and len(fake.calls) == 3


async def test_timeout_no_es_connection(make_openai, request_):
    c = make_openai(max_retries=0)
    install(c, Fake(openai.APITimeoutError(request=request_)))
    r = await c.generate(MSGS)
    assert r.error.type == "timeout" and r.error.retryable


async def test_error_5xx_y_desconocido(make_openai, make_status_error):
    c = make_openai(max_retries=0)
    install(c, Fake(make_status_error(openai, "InternalServerError", 500), RuntimeError("boom")))
    assert (await c.generate(MSGS)).error.type == "server"
    r = await c.generate(MSGS)
    assert r.error.type == "unknown" and not r.error.retryable


# ---- streaming ----
def chunk(text):
    return NS(choices=[NS(delta=NS(content=text))] if text is not None else [])


class agen:
    """Imita `AsyncStream`: iterable asíncrono y context manager que registra el cierre."""

    def __init__(self, items, fail_with=None):
        self.items, self.fail_with, self.closed = items, fail_with, False

    async def __aiter__(self):
        for i in self.items:
            yield i
        if self.fail_with:
            raise self.fail_with

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        self.closed = True


class StreamFake(Fake):
    async def __call__(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


async def test_stream_orden_y_chunk_de_uso(make_openai):
    c = make_openai()
    fake = install(c, StreamFake(agen([chunk("a"), chunk(""), chunk("b"), chunk(None)])))
    assert [t async for t in c.stream(MSGS)] == ["a", "b"]
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["stream_options"] == {"include_usage": True}


async def test_stream_reintenta_antes_del_primer_chunk(make_openai, request_):
    c = make_openai()
    fake = install(c, StreamFake(openai.APIConnectionError(request=request_), agen([chunk("x")])))
    assert [t async for t in c.stream(MSGS)] == ["x"]
    assert len(fake.calls) == 2


async def test_stream_falla_tras_chunk_sin_reintento(make_openai, request_):
    c = make_openai()
    fake = install(c, StreamFake(agen([chunk("x")], fail_with=openai.APIConnectionError(request=request_))))
    got = []
    with pytest.raises(LLMClientError) as ei:
        async for t in c.stream(MSGS):
            got.append(t)
    assert got == ["x"] and ei.value.error.type == "connection" and len(fake.calls) == 1


async def test_stream_error_se_convierte(make_openai, make_status_error):
    c = make_openai()
    install(c, StreamFake(make_status_error(openai, "AuthenticationError", 401)))
    with pytest.raises(LLMClientError) as ei:
        [t async for t in c.stream(MSGS)]
    assert ei.value.error.type == "authentication"


async def test_llamadas_concurrentes_no_bloquean(make_openai):
    c = make_openai()

    async def slow(**kw):
        await asyncio.sleep(0.2)
        return completion()

    install(c, slow)
    t0 = time.perf_counter()
    rs = await asyncio.gather(c.generate(MSGS), c.generate(MSGS))
    assert all(r.ok for r in rs) and time.perf_counter() - t0 < 0.35


async def test_concurrencia_no_bloquea_otra_vez(make_openai):
    c = make_openai()

    async def slow(**kw):
        await asyncio.sleep(0.3)
        return completion()

    install(c, slow)
    t0 = time.perf_counter()
    await asyncio.gather(*(c.generate(MSGS) for _ in range(3)))
    assert time.perf_counter() - t0 < 0.55  # en serie serían 0.9s


def test_base_es_abstracta():
    with pytest.raises(TypeError):
        BaseLLMClient(None)


async def test_cancelled_error_se_propaga(make_openai):
    c = make_openai()
    install(c, Fake(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await c.generate(MSGS)


async def test_stream_entrada_vacia_lanza(make_openai):
    with pytest.raises(LLMClientError) as ei:
        [t async for t in make_openai().stream([])]
    assert ei.value.error.type == "bad_request"


async def test_backoff_exponencial(make_openai, make_status_error, monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda a, b: 0)
    c = make_openai(max_retries=3)
    delays = []

    async def rec(d):
        delays.append(d)

    c._sleep = rec
    install(c, Fake(*[make_status_error(openai, "InternalServerError", 500) for _ in range(4)]))
    r = await c.generate(MSGS)
    assert r.attempts == 4 and delays == [0.5, 1.0, 2.0]


async def test_retry_after_se_respeta_y_se_limita(make_openai, request_):
    import httpx2

    def err(value):
        resp = httpx2.Response(429, request=request_, headers={"retry-after": value})
        return openai.RateLimitError("x", response=resp, body=None)

    c = make_openai(max_retries=2)
    delays = []

    async def rec(d):
        delays.append(d)

    c._sleep = rec
    install(c, Fake(err("3"), err("999"), completion()))
    assert (await c.generate(MSGS)).ok
    assert delays == [3.0, 20.0]


@pytest.mark.parametrize("status,kind", [(408, "timeout"), (409, "unknown")])
async def test_408_409_reintentables(make_openai, make_status_error, status, kind):
    c = make_openai()
    cls = "APIStatusError"
    install(c, Fake(make_status_error(openai, cls, status), completion()))
    r = await c.generate(MSGS)
    assert r.ok and r.attempts == 2


async def test_mensaje_redacta_clave(make_openai, make_status_error):
    c = make_openai()
    err = make_status_error(openai, "AuthenticationError", 401)
    err.args = ("Incorrect API key provided: sk-proj-****abcd. Find it at ...",)
    install(c, Fake(err))
    r = await c.generate(MSGS)
    assert "sk-proj" not in r.error.message and "sk-***" in r.error.message


async def test_error_a_mitad_de_stream_se_reintenta(make_openai):
    c = make_openai()
    bad = openai.APIError("boom", request=None, body={"type": "server_error", "message": "x"})
    fake = install(c, StreamFake(agen([], fail_with=bad), agen([chunk("ok")])))
    assert [t async for t in c.stream(MSGS)] == ["ok"] and len(fake.calls) == 2


async def test_stream_cierra_el_stream_al_cortar(make_openai):
    c = make_openai()
    s = agen([chunk("a"), chunk("b")])
    install(c, StreamFake(s))
    gen = c.stream(MSGS)
    async for _ in gen:
        break
    await gen.aclose()
    assert s.closed
