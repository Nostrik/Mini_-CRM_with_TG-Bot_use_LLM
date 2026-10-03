"""Тесты OpenRouterClient. Реальных запросов в сеть нет: транспорт httpx подменён на MockTransport."""
import json
from typing import Callable, Optional

import httpx
import pytest
from pydantic import BaseModel

import openrouter_api
from openrouter_api import BASE_URL, OpenRouterClient, OpenRouterError


# ---------- вспомогательное ----------
def ok_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


class LeadExtract(BaseModel):
    name: Optional[str] = None
    contact: Optional[str] = None
    request: Optional[str] = None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Ретраи не должны реально ждать."""
    async def fake_sleep(_):
        return None
    monkeypatch.setattr(openrouter_api.asyncio, "sleep", fake_sleep)


@pytest.fixture
def make_client():
    """Фабрика клиента, у которого все HTTP-запросы идут в handler. Возвращает (client, calls)."""
    created = []

    async def _make(handler: Callable[[httpx.Request], httpx.Response], **kwargs):
        calls: list[httpx.Request] = []

        def wrapped(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return handler(request)

        kwargs.setdefault("fallback_models", [])  # не зависим от OPENROUTER_FALLBACK_MODELS в .env
        client = OpenRouterClient(api_key="test-key", **kwargs)
        headers = client._http.headers
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url=BASE_URL, headers=headers, transport=httpx.MockTransport(wrapped)
        )
        created.append(client)
        return client, calls

    yield _make


# ---------- инициализация ----------
def test_init_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError):
        OpenRouterClient()


async def test_init_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    client = OpenRouterClient()
    assert client.api_key == "env-key"
    await client.close()


def test_default_model_is_free_router():
    assert openrouter_api.DEFAULT_MODEL.endswith("free")


# ---------- chat ----------
async def test_chat_returns_text_and_sends_payload(make_client):
    client, calls = await make_client(lambda r: ok_response("привет"))
    result = await client.chat([{"role": "user", "content": "hi"}], max_tokens=50)

    assert result == "привет"
    assert len(calls) == 1
    req = calls[0]
    assert req.url.path.endswith("/chat/completions")
    assert req.headers["Authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert body["model"] == client.model
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 50
    assert "models" not in body


async def test_chat_custom_model_overrides_default(make_client):
    client, calls = await make_client(lambda r: ok_response("x"))
    await client.chat([{"role": "user", "content": "hi"}], model="some/model:free")
    assert json.loads(calls[0].content)["model"] == "some/model:free"


async def test_chat_adds_fallback_models(make_client):
    client, calls = await make_client(
        lambda r: ok_response("x"), model="main/model", fallback_models=["b/model", "c/model"]
    )
    await client.chat([{"role": "user", "content": "hi"}])
    assert json.loads(calls[0].content)["models"] == ["main/model", "b/model", "c/model"]


async def test_chat_null_content_becomes_empty_string(make_client):
    client, _ = await make_client(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    )
    assert await client.chat([{"role": "user", "content": "hi"}]) == ""


async def test_chat_unexpected_format_raises(make_client):
    client, _ = await make_client(lambda r: httpx.Response(200, json={"foo": "bar"}))
    with pytest.raises(OpenRouterError):
        await client.chat([{"role": "user", "content": "hi"}])


# ---------- ошибки и ретраи ----------
async def test_retries_on_429_then_succeeds(make_client):
    responses = [httpx.Response(429, text="slow down"), ok_response("ok")]
    client, calls = await make_client(lambda r: responses.pop(0))
    assert await client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(calls) == 2


async def test_retries_on_500_then_succeeds(make_client):
    responses = [httpx.Response(503), httpx.Response(502), ok_response("ok")]
    client, calls = await make_client(lambda r: responses.pop(0))
    assert await client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(calls) == 3


async def test_gives_up_after_max_retries(make_client):
    client, calls = await make_client(lambda r: httpx.Response(500), max_retries=3)
    with pytest.raises(OpenRouterError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 3


async def test_4xx_is_not_retried(make_client):
    client, calls = await make_client(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(OpenRouterError, match="401"):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


async def test_error_field_in_200_body_raises(make_client):
    client, _ = await make_client(
        lambda r: httpx.Response(200, json={"error": {"message": "provider down"}})
    )
    with pytest.raises(OpenRouterError, match="provider down"):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_network_error_is_retried_and_raised(make_client):
    def boom(request):
        raise httpx.ConnectError("no network")

    client, calls = await make_client(boom, max_retries=2)
    with pytest.raises(OpenRouterError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2


# ---------- _parse_json ----------
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Вот результат: {"a": 1} Надеюсь, помогло.',
    ],
)
def test_parse_json_variants(text):
    assert OpenRouterClient._parse_json(text) == {"a": 1}


def test_parse_json_invalid_raises():
    with pytest.raises(OpenRouterError):
        OpenRouterClient._parse_json("совсем не json")


# ---------- chat_json ----------
async def test_chat_json_parses_into_schema(make_client):
    payload = '{"name": "Ирина", "contact": "@irina_coffee", "request": "сайт для кофейни"}'
    client, calls = await make_client(lambda r: ok_response(payload))

    lead = await client.chat_json("system prompt", "сообщение", LeadExtract)

    assert isinstance(lead, LeadExtract)
    assert lead.name == "Ирина"
    assert lead.contact == "@irina_coffee"
    body = json.loads(calls[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "LeadExtract"
    assert body["messages"][0] == {"role": "system", "content": "system prompt"}
    assert body["messages"][1] == {"role": "user", "content": "сообщение"}


async def test_chat_json_handles_fenced_output(make_client):
    client, _ = await make_client(lambda r: ok_response('```json\n{"name": "Олег"}\n```'))
    lead = await client.chat_json("sys", "msg", LeadExtract)
    assert lead.name == "Олег"
    assert lead.contact is None


async def test_chat_json_falls_back_without_response_format(make_client):
    responses = [httpx.Response(400, text="json_schema not supported"), ok_response('{"name": "Анна"}')]
    client, calls = await make_client(lambda r: responses.pop(0))

    lead = await client.chat_json("sys", "msg", LeadExtract)

    assert lead.name == "Анна"
    assert len(calls) == 2
    second = json.loads(calls[1].content)
    assert "response_format" not in second
    assert "JSON" in second["messages"][0]["content"]
    assert "name" in second["messages"][0]["content"]  # схема попала в промпт


async def test_chat_json_falls_back_on_invalid_json(make_client):
    responses = [ok_response("не json вообще"), ok_response('{"name": "Пётр"}')]
    client, calls = await make_client(lambda r: responses.pop(0))
    lead = await client.chat_json("sys", "msg", LeadExtract)
    assert lead.name == "Пётр"
    assert len(calls) == 2


async def test_chat_json_raises_if_fallback_also_fails(make_client):
    client, _ = await make_client(lambda r: ok_response("мусор"))
    with pytest.raises(OpenRouterError):
        await client.chat_json("sys", "msg", LeadExtract)


# ---------- контекстный менеджер ----------
async def test_async_context_manager_closes_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    async with OpenRouterClient() as client:
        assert not client._http.is_closed
    assert client._http.is_closed
