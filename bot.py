"""Telegram-бот для сбора заявок в мини-CRM.

Поток: клиент пишет боту -> extract_lead() достаёт имя/контакт/запрос/теги ->
если чего-то не хватает, бот задаёт уточняющий вопрос -> готовая заявка уходит в БД (db.create_lead).

Переменные окружения (.env):
    TELEGRAM_BOT_TOKEN  - токен бота от @BotFather
    OPENROUTER_API_KEY  - ключ OpenRouter (читается в openrouter_api.py)
    ADMIN_CHAT_ID       - (необязательно) ваш chat_id: туда бот пришлёт уведомления
                          «запущен» и «остановлен». Узнать id: напишите боту /id

Ожидаемый интерфейс модуля db.py (его нужно написать отдельно):
    async def init_db() -> None
    async def create_lead(draft: LeadDraft, source: str,
                          telegram_id: int | None = None,
                          telegram_username: str | None = None) -> int   # возвращает id лида
Поля name / contact / request в БД должны допускать NULL: если клиент упорно не отвечает
на вопрос, заявка всё равно сохраняется, а полный текст переписки лежит в draft.raw_text.

Запуск: python bot.py
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message
from dotenv import load_dotenv

load_dotenv()  # подтягиваем .env до чтения токена

import db  # noqa: E402  (модуль с БД, см. интерфейс в докстринге)
from extract import (  # noqa: E402
    MAX_NAME_LEN,
    LeadDraft,
    extract_lead,
    next_missing_field,
    next_question,
)
from logger import AppLogger  # noqa: E402
from openrouter_api import OpenRouterClient  # noqa: E402

log = AppLogger()

# Сколько раз бот переспрашивает одно и то же поле, прежде чем сохранить заявку как есть
MAX_ASKS_PER_FIELD = 3
# Ограничение длины входящего текста (защита от огромных сообщений и расхода токенов)
MAX_INPUT_LEN = 2000

# Момент запуска процесса: нужен команде /status (аптайм) и уведомлению администратору
START_MONOTONIC = time.monotonic()
STARTED_AT = datetime.now()

GREETING = (
    "Здравствуйте! Я помогу оставить заявку в агентство.\n\n"
    "Опишите задачу и оставьте контакт для связи (телефон, email или @username). "
    "Если чего-то не хватит, я уточню."
)
HELP_TEXT = (
    "Просто напишите, что вам нужно, и как с вами связаться. "
    "Команды: /start - начать заново, /cancel - отменить заявку, "
    "/status - проверить, что бот онлайн."
)

router = Router()


# ---------- состояние диалога ----------
@dataclass
class Session:
    """Состояние одного диалога. Хранится в памяти (при перезапуске бота сбрасывается)."""

    draft: LeadDraft = field(default_factory=LeadDraft)
    asking: Optional[str] = None  # о каком поле бот спросил в прошлый раз
    asks: dict[str, int] = field(default_factory=dict)  # сколько раз спрашивали каждое поле
    llm_failed_any: bool = False  # LLM хоть раз не сработала в этой заявке


_sessions: dict[int, Session] = {}
# По одной блокировке на чат: если клиент шлёт сообщения пачкой, обрабатываем по очереди
_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def _reset(chat_id: int) -> None:
    _sessions.pop(chat_id, None)


def _apply_degraded_fallback(draft: LeadDraft, text: str, asking: Optional[str]) -> LeadDraft:
    """Если LLM не сработала на этом сообщении, а бот ждал имя или запрос,
    берём ответ клиента как есть. Иначе бот зациклится на одном и том же вопросе."""
    if not draft.llm_failed or asking not in ("name", "request"):
        return draft
    if getattr(draft, asking):
        return draft
    value = " ".join(text.split())
    if asking == "name":
        value = value[:MAX_NAME_LEN]
    setattr(draft, asking, value)
    log.warning(f"LLM недоступна: поле '{asking}' заполнено ответом клиента без обработки")
    return draft


# ---------- сохранение заявки ----------
async def _finish(message: Message, session: Session, forced: bool) -> bool:
    """Сохраняет заявку в БД и отвечает клиенту. Возвращает True при успехе."""
    draft = session.draft
    user = message.from_user

    if forced and not draft.contact and user and user.username:
        # клиент так и не дал контакт: хотя бы свяжемся через Telegram
        draft.contact = f"@{user.username}"
        log.info("Контакт не получен, использован Telegram-username клиента")

    try:
        lead_id = await db.create_lead(
            draft,
            source="bot",
            telegram_id=user.id if user else None,
            telegram_username=user.username if user else None,
        )
    except Exception:
        log.error("Не удалось сохранить заявку в БД", exc_info=True)
        await message.answer(
            "Не получилось сохранить заявку. Попробуйте, пожалуйста, написать ещё раз через минуту."
        )
        return False

    log.info(
        f"Заявка #{lead_id} сохранена: chat_id={message.chat.id}, "
        f"принудительно={forced}, не хватает={draft.missing or '-'}, "
        f"теги={draft.tags or '-'}, llm_сбой={session.llm_failed_any}"
    )
    greeting = f"Спасибо, {draft.name}!" if draft.name else "Спасибо!"
    await message.answer(f"{greeting} Заявка №{lead_id} принята, мы свяжемся с вами в ближайшее время.")
    _reset(message.chat.id)
    return True


# ---------- обработчики ----------
@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    _reset(message.chat.id)
    log.info(f"/start: chat_id={message.chat.id}")
    await message.answer(GREETING)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message) -> None:
    _reset(message.chat.id)
    log.info(f"/cancel: chat_id={message.chat.id}")
    await message.answer("Заявка отменена. Если захотите начать заново, просто напишите.")


def _format_uptime(seconds: float) -> str:
    """Секунды в читаемый вид: '2 д 3 ч 15 мин 7 с'."""
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} д")
    if days or hours:
        parts.append(f"{hours} ч")
    if days or hours or minutes:
        parts.append(f"{minutes} мин")
    parts.append(f"{secs} с")
    return " ".join(parts)


@router.message(Command("status", "ping"))
async def cmd_status(message: Message) -> None:
    """Проверка, что бот жив: аптайм, время запуска, число незавершённых диалогов."""
    uptime = _format_uptime(time.monotonic() - START_MONOTONIC)
    log.info(f"/status: chat_id={message.chat.id}")
    await message.answer(
        "✅ Бот онлайн\n"
        f"Работает: {uptime}\n"
        f"Запущен: {STARTED_AT:%d.%m.%Y %H:%M:%S}\n"
        f"Диалогов в работе: {len(_sessions)}"
    )


@router.message(Command("id"))
async def cmd_id(message: Message) -> None:
    """Показывает chat_id: его нужно вписать в ADMIN_CHAT_ID для уведомлений о запуске."""
    await message.answer(f"Ваш chat_id: {message.chat.id}")


@router.message(F.text, ~F.text.startswith("/"))
async def handle_text(message: Message, llm: OpenRouterClient) -> None:
    chat_id = message.chat.id
    text = message.text[:MAX_INPUT_LEN]

    async with _locks[chat_id]:
        session = _sessions.setdefault(chat_id, Session())
        log.info(f"Сообщение: chat_id={chat_id}, длина={len(text)}, ждём поле={session.asking or '-'}")

        try:
            await message.bot.send_chat_action(chat_id, ChatAction.TYPING)

            # сбрасываем флаг, чтобы он отражал сбой именно этого сообщения
            session.draft.llm_failed = False
            draft = await extract_lead(llm, text, session.draft, session.asking)
            draft = _apply_degraded_fallback(draft, text, session.asking)
            session.llm_failed_any = session.llm_failed_any or draft.llm_failed
            session.draft = draft

            missing_field = next_missing_field(draft)

            # всё собрано
            if missing_field is None:
                await _finish(message, session, forced=False)
                return

            # клиент не отвечает на вопрос слишком долго: сохраняем что есть
            if session.asks.get(missing_field, 0) >= MAX_ASKS_PER_FIELD:
                log.warning(f"Поле '{missing_field}' не получено за {MAX_ASKS_PER_FIELD} вопроса, сохраняю как есть")
                await _finish(message, session, forced=True)
                return

            # задаём уточняющий вопрос
            session.asks[missing_field] = session.asks.get(missing_field, 0) + 1
            session.asking = missing_field
            await message.answer(next_question(draft))

        except Exception:
            log.error(f"Ошибка обработки сообщения: chat_id={chat_id}", exc_info=True)
            await message.answer("Произошла ошибка. Попробуйте, пожалуйста, ещё раз.")


@router.message()
async def handle_other(message: Message) -> None:
    """Фото, голосовые, стикеры и прочее: пока понимаем только текст."""
    log.info(f"Нетекстовое сообщение: chat_id={message.chat.id}, тип={message.content_type}")
    await message.answer("Пока я понимаю только текст. Напишите, пожалуйста, задачу и контакт сообщением.")


# ---------- уведомления и меню команд ----------
def _parse_admin_chat_id(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        log.warning("ADMIN_CHAT_ID должен быть числом, уведомления администратору отключены")
        return None


async def _notify_admin(bot: Bot, chat_id: Optional[int], text: str) -> None:
    """Пишет администратору в Telegram. Любая ошибка только логируется и работу бота не ломает."""
    if chat_id is None:
        return
    try:
        await bot.send_message(chat_id, text)
    except Exception as e:
        log.warning(
            f"Не удалось отправить уведомление администратору "
            f"(нужно хотя бы раз написать боту /start): {e}"
        )


async def _setup_commands(bot: Bot) -> None:
    """Меню команд в Telegram (кнопка «Меню» рядом с полем ввода)."""
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Начать заново"),
                BotCommand(command="status", description="Проверить, что бот онлайн"),
                BotCommand(command="cancel", description="Отменить заявку"),
                BotCommand(command="help", description="Помощь"),
            ]
        )
    except Exception as e:
        log.warning(f"Не удалось установить меню команд: {e}")


# ---------- запуск ----------
async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        log.error("Не задан TELEGRAM_BOT_TOKEN (добавьте его в .env)")
        raise SystemExit(1)

    await db.init_db()
    llm = OpenRouterClient()  # один клиент на весь процесс
    bot = Bot(token=token)
    dp = Dispatcher(llm=llm)  # llm автоматически передаётся в хендлеры по имени параметра
    dp.include_router(router)

    admin_chat_id = _parse_admin_chat_id(os.getenv("ADMIN_CHAT_ID"))

    try:
        me = await bot.get_me()
        log.info(f"Бот запущен: @{me.username}")
        await _setup_commands(bot)
        await _notify_admin(bot, admin_chat_id, f"🟢 Бот @{me.username} запущен ({STARTED_AT:%d.%m.%Y %H:%M:%S})")
        await dp.start_polling(bot)
    finally:
        log.info("Остановка бота")
        await _notify_admin(bot, admin_chat_id, "🔴 Бот остановлен")
        await llm.close()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Бот остановлен")
