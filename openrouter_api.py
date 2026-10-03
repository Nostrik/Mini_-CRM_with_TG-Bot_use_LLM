"""Клиент для OpenRouter (OpenAI-совместимый API) с логированием через AppLogger.

Зависимости: pip install httpx pydantic python-dotenv colorlog
Переменные окружения (можно в .env):
    OPENROUTER_API_KEY  - обязательный ключ
    OPENROUTER_MODEL    - основная модель (необязательно, по умолчанию бесплатный роутер)
    OPENROUTER_FALLBACK_MODELS - запасные модели через запятую (необязательно). OpenRouter
                          переключится на них, если основная недоступна (лимит, сбой провайдера).
                          Всего в запросе не больше 3 моделей: основная + 2 запасные.

Что и на каком уровне логируется:
    DEBUG   - детали запросов (модель, число сообщений, обрезанный текст), разбор JSON
    INFO    - создание клиента, успешные ответы (модель, токены, время), закрытие клиента
    WARNING - повторные попытки, откат chat_json на запасной режим
    ERROR   - исчерпанные попытки, 4xx, неверный формат ответа, невалидный JSON
Ключ API и полные тексты пользователей на уровнях INFO и выше не пишутся.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Optional, Type

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from logger import AppLogger

load_dotenv()  # подтягивает переменные из .env в os.environ

log = AppLogger()

BASE_URL = "https://openrouter.ai/api/v1"
# Бесплатный авто-роутер OpenRouter: сам выбирает доступную :free модель.
# Список бесплатных моделей часто меняется, поэтому жёстко прописанные id быстро устаревают.
# Лимиты free: 20 запросов/мин, 200/день (1000/день после пополнения баланса на $10+).
# Если нужна конкретная модель, возьмите id с суффиксом ":free" на openrouter.ai/models
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# OpenRouter принимает в одном запросе не больше 3 моделей (основная + запасные)
MAX_ROUTE_MODELS = 3


def _env_list(name: str) -> list[str]:
    """Список из переменной окружения через запятую: 'a, b ,c' -> ['a', 'b', 'c']."""
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


DEFAULT_FALLBACK_MODELS = _env_list("OPENROUTER_FALLBACK_MODELS")

# Коды ответа, при которых имеет смысл повторить запрос
RETRY_STATUSES = {429, 500, 502, 503, 504}
# Сколько символов пользовательского текста показывать в DEBUG-логах
PREVIEW_LEN = 100


class OpenRouterError(Exception):
    """Ошибка обращения к OpenRouter."""


def _preview(text: str, limit: int = PREVIEW_LEN) -> str:
    """Короткая версия текста для логов (в одну строку, с обрезкой)."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        fallback_models: Optional[list[str]] = None,
        app_url: str = "",
        app_name: str = "mini-crm",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            log.error("Не задан OPENROUTER_API_KEY (проверьте .env и имя переменной)")
            raise OpenRouterError("Не задан OPENROUTER_API_KEY")
        self.model = model
        candidates = DEFAULT_FALLBACK_MODELS if fallback_models is None else fallback_models
        # без дублей и без основной модели; всего в запросе не больше MAX_ROUTE_MODELS
        unique = [m for i, m in enumerate(candidates) if m != model and m not in candidates[:i]]
        if len(unique) > MAX_ROUTE_MODELS - 1:
            log.warning(
                f"Запасных моделей слишком много: используются первые {MAX_ROUTE_MODELS - 1}, "
                f"остальные игнорируются: {unique[MAX_ROUTE_MODELS - 1:]}"
            )
        self.fallback_models = unique[: MAX_ROUTE_MODELS - 1]
        self.max_retries = max_retries
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": app_name,  # необязательные заголовки для статистики OpenRouter
        }
        if app_url:
            headers["HTTP-Referer"] = app_url
        self._http = httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=timeout)
        log.info(
            f"OpenRouter-клиент создан: model={self.model}, "
            f"fallback={self.fallback_models or '-'}, timeout={timeout}s, retries={max_retries}"
        )

    # ---------- низкий уровень ----------
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST с повторными попытками. 4xx (кроме 429) не повторяются."""
        last_err = "неизвестная ошибка"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._http.post(path, json=payload)
            except httpx.TransportError as e:
                last_err = f"сетевая ошибка: {e!r}"
            else:
                if resp.status_code in RETRY_STATUSES:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                elif resp.status_code >= 400:
                    # 401 (ключ), 402 (баланс), 400 (плохой запрос) и т.п. повторять бессмысленно
                    log.error(f"OpenRouter вернул HTTP {resp.status_code}: {resp.text[:500]}")
                    raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                else:
                    try:
                        data = resp.json()
                    except ValueError:
                        last_err = f"ответ не является JSON: {resp.text[:200]}"
                    else:
                        if "error" in data:
                            last_err = f"ошибка провайдера: {data['error']}"
                        else:
                            return data

            if attempt < self.max_retries:
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                log.warning(
                    f"Попытка {attempt}/{self.max_retries} не удалась ({last_err}). "
                    f"Повтор через {delay}с"
                )
                await asyncio.sleep(delay)
            else:
                log.warning(f"Попытка {attempt}/{self.max_retries} не удалась ({last_err})")

        log.error(f"Запрос к OpenRouter не удался после {self.max_retries} попыток: {last_err}")
        raise OpenRouterError(f"Запрос не удался после {self.max_retries} попыток: {last_err}")

    # ---------- публичный API ----------
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **extra: Any,
    ) -> str:
        """Обычный чат-запрос, возвращает текст ответа."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        if self.fallback_models:
            payload["models"] = [payload["model"], *self.fallback_models]

        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        log.debug(
            f"chat -> model={payload['model']}, сообщений={len(messages)}, "
            f"max_tokens={max_tokens}, extra={list(extra)}, user='{_preview(last_user)}'"
        )

        started = time.perf_counter()
        data = await self._post("/chat/completions", payload)
        elapsed = time.perf_counter() - started

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            log.error(f"Неожиданный формат ответа OpenRouter: {str(data)[:300]}")
            raise OpenRouterError(f"Неожиданный формат ответа: {data}") from e

        usage = data.get("usage") or {}
        log.info(
            f"chat OK: model={data.get('model', payload['model'])}, {elapsed:.2f}с, "
            f"токены prompt={usage.get('prompt_tokens', '?')} "
            f"completion={usage.get('completion_tokens', '?')}"
        )
        if not content:
            log.warning("Модель вернула пустой ответ")
        log.debug(f"chat <- '{_preview(content, 300)}'")
        return content

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: Type[BaseModel],
        model: Optional[str] = None,
    ) -> BaseModel:
        """Структурированное извлечение: возвращает экземпляр pydantic-схемы."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
        log.debug(f"chat_json: схема={schema.__name__}, режим=json_schema")
        try:
            text = await self.chat(messages, model=model, response_format=response_format)
            result = schema.model_validate(self._parse_json(text))
            log.info(f"chat_json OK: {schema.__name__} (json_schema)")
            return result
        except (OpenRouterError, ValueError) as e:
            # Не все бесплатные модели умеют json_schema: повторяем без него,
            # схему кладём прямо в промпт.
            log.warning(
                f"chat_json: режим json_schema не сработал ({_preview(str(e), 200)}). "
                f"Пробую запасной режим: схема в промпте"
            )
            messages[0]["content"] = (
                f"{system}\n\nВерни ТОЛЬКО валидный JSON по схеме, без пояснений:\n{schema_json}"
            )
            text = await self.chat(messages, model=model)
            try:
                result = schema.model_validate(self._parse_json(text))
            except ValueError as e2:
                log.error(
                    f"chat_json: запасной режим тоже не дал валидный {schema.__name__}: "
                    f"{_preview(str(e2), 200)}"
                )
                raise
            log.info(f"chat_json OK: {schema.__name__} (запасной режим)")
            return result

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Достаёт JSON даже если модель обернула его в ```json ... ```."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                log.debug("JSON извлечён из текста с лишним обрамлением")
                return json.loads(m.group(0))
            log.error(f"Модель вернула не JSON: '{_preview(text, 200)}'")
            raise OpenRouterError(f"Модель вернула не JSON: {text[:200]}")

    async def close(self) -> None:
        await self._http.aclose()
        log.info("OpenRouter-клиент закрыт")

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


# ---------- пример использования ----------
if __name__ == "__main__":
    from typing import Optional as Opt

    class LeadExtract(BaseModel):
        name: Opt[str] = None
        contact: Opt[str] = None
        request: Opt[str] = None
        tags: list[str] = []
        missing: list[str] = []

    async def demo():
        async with OpenRouterClient() as llm:
            res = await llm.chat_json(
                system=(
                    "Извлеки данные заявки из сообщения. Если поля нет в тексте, верни null. "
                    "Не выдумывай. В missing перечисли отсутствующие поля (name, contact, request)."
                ),
                user="Привет, я Ирина, нужен сайт для кофейни, пишите на @irina_coffee",
                schema=LeadExtract,
            )
            print(res.model_dump())

    asyncio.run(demo())