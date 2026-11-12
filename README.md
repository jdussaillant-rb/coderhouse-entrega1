# Pre-entrega 1: Cliente de LLM robusto y asíncrono

Cliente asíncrono e intercambiable para **OpenAI** y **Anthropic**, con validación Pydantic,
streaming, reintentos con backoff exponencial y errores estructurados.

## Estructura

```
llm_client/
  schemas.py          # ChatMessage, ModelConfig, ModelResponse, LLMError...
  base.py             # BaseLLMClient (ABC): generate(), stream(), reintentos
  openai_client.py    # OpenAIClient (AsyncOpenAI)
  anthropic_client.py # AnthropicClient (AsyncAnthropic)
  manager.py          # AsyncLLMManager: elige el proveedor por configuración
  errors.py           # LLMClientError
main.py               # demo: modo normal + streaming
tests/                # pruebas offline (sin red ni API keys)
```

## Configuración del entorno (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # solo ejecución: requirements.txt
cp .env.example .env               # y completa tus claves
```

### Variables de entorno

| Variable | Descripción | Por defecto |
|---|---|---|
| `OPENAI_API_KEY` | Clave de OpenAI (necesaria para usar OpenAI) | - |
| `ANTHROPIC_API_KEY` | Clave de Anthropic (necesaria para usar Anthropic) | - |
| `LLM_PROVIDER` | `openai` o `anthropic` | `main.py`: todos los que tengan clave; `AsyncLLMManager.from_env()`: `openai` |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | Modelo a usar | `gpt-5-mini` / `claude-opus-5-5` |
| `LLM_TEMPERATURE` | 0 a 2 (opcional; Anthropic la ignora) | no se envía |
| `LLM_MAX_TOKENS` | Máximo de tokens de salida | `4096` |
| `LLM_TIMEOUT` | Timeout en segundos | `60` |
| `LLM_MAX_RETRIES` | Reintentos ante errores transitorios | `2` |

## Ejecución

```bash
python main.py                          # todos los proveedores con clave (si LLM_PROVIDER no está definida)
python main.py --provider anthropic
python main.py --provider openai --prompt "¿Qué es la entropía?"
python main.py --directo                # usa OpenAIClient/AnthropicClient sin el manager
python main.py --demo-errores           # clave inválida a propósito (hace una llamada real)
```

Si falta la clave de un proveedor, se muestra un mensaje y se continúa. El código de salida es
`0`, salvo que se haya intentado algún proveedor y todos hayan fallado (`1`), o que
`--provider`/`LLM_PROVIDER` sea desconocido (`2`).

## Pruebas

```bash
python -m pytest -q
```

Son offline: reemplazan los métodos de los SDK por fakes, no usan red ni `.env`.

## Uso como librería

```python
from llm_client import AsyncLLMManager

async with AsyncLLMManager.from_env("anthropic") as llm:
    resp = await llm.generate("¿Qué es la entropía?")   # nunca lanza por errores de API
    print(resp.content if resp.ok else resp.error)

    async for token in llm.stream("¿Qué es la entropía?"):  # lanza LLMClientError si falla
        print(token, end="", flush=True)
```

## Decisiones de diseño

- **Interfaz común**: `BaseLLMClient` es una clase abstracta (ABC, no se puede instanciar).
  `generate()` y `stream()` son la interfaz pública y están implementadas **una sola vez** en la
  base (patrón *template method*): validan la entrada, reintentan y traducen errores. Cada
  proveedor solo implementa los métodos abstractos `_generate_once()`, `_stream_once()`,
  `_map_error()` y `aclose()`. Así `OpenAIClient` y `AnthropicClient` son intercambiables.

- **Asincronía**: solo `AsyncOpenAI` y `AsyncAnthropic`; los SDK se crean con `max_retries=0`
  y los reintentos los hace `BaseLLMClient` (backoff exponencial con jitter y `asyncio.sleep`).
- **Errores**: autenticación, permisos, 400/404/422 no se reintentan; rate limit (429), timeout,
  conexión, 408, 409 y 5xx/529 sí, hasta `max_retries`; si el servidor envía `Retry-After` se respeta (máx. 20 s).
  Los errores que llegan a mitad de stream (evento `error`) se clasifican por el tipo del cuerpo.
  Los mensajes de error se redactan para no filtrar claves (`sk-***`). Un 429 con `insufficient_quota` (sin saldo) no se reintenta.
- **`generate()`** devuelve un `ModelResponse` con `error` poblado; **`stream()`** lanza
  `LLMClientError` (con `.error` estructurado, también ante entrada inválida), porque un generador no puede devolver un error.
  En streaming solo se reintenta si aún no se emitió ningún token.
- **API key ausente**: el constructor lanza `ValueError` claro (en ambos proveedores).
- **Anthropic**: el `system` se envía aparte; `temperature` no se envía (los modelos actuales
  no aceptan parámetros de muestreo) y se avisa por log.
