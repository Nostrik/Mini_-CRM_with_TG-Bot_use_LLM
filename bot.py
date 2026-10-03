"""Telegram-бот для сбора заявок в мини-CRM.

Логика:
    клиент может прислать всю заявку одним сообщением;
    LLM извлекает всю доступную информацию;
    если обязательных данных достаточно — заявка сразу сохраняется;
    если данных не хватает — LLM формирует один конкретный вопрос;
    ответ клиента снова анализируется вместе со всем накопленным контекстом.

Бот НЕ ведёт клиента по фиксированному сценарию:
    имя -> контакт -> запрос.

Переменные окружения:
    TELEGRAM_BOT_TOKEN
    OPENROUTER_API_KEY
    OPENROUTER_MODEL
    OPENROUTER_FALLBACK_MODELS
    ADMIN_CHAT_ID
    DB_PATH

Запуск:
    python bot.py
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

load_dotenv()

import db  # noqa: E402

from extract import (  # noqa: E402
    LeadDraft,
    extract_lead,
)

from logger import AppLogger  # noqa: E402
from openrouter_api import OpenRouterClient  # noqa: E402


log = AppLogger()


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MAX_INPUT_LEN = 2000

START_MONOTONIC = time.monotonic()
STARTED_AT = datetime.now()

GREETING = (
    "Здравствуйте! Я помогу оставить заявку в агентство.\n\n"
    "Опишите задачу и оставьте контакт для связи "
    "(телефон, email или @username).\n"
    "Можно написать всё одним сообщением — "
    "если чего-то будет не хватать, я уточню."
)

HELP_TEXT = (
    "Просто напишите, что вам нужно и как с вами связаться. "
    "Можно указать всю информацию одним сообщением.\n\n"
    "Команды:\n"
    "/start — начать заново\n"
    "/cancel — отменить заявку\n"
    "/status — проверить, что бот онлайн."
)


router = Router()


# ---------------------------------------------------------------------------
# Состояние диалога
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Состояние одного диалога.

    Хранится в памяти.
    При перезапуске бота незавершённые диалоги сбрасываются.
    """

    draft: LeadDraft = field(default_factory=LeadDraft)

    # Последний вопрос, который бот реально показал клиенту.
    #
    # Он нужен LLM как контекст.
    # Само состояние «какое поле спрашивать дальше»
    # больше нигде не хранится.
    last_question: Optional[str] = None

    # Была ли LLM недоступна хотя бы один раз.
    llm_failed_any: bool = False


_sessions: dict[int, Session] = {}

# Если клиент быстро отправляет несколько сообщений подряд,
# обрабатываем их последовательно.
_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def _reset(chat_id: int) -> None:
    _sessions.pop(chat_id, None)


# ---------------------------------------------------------------------------
# Сохранение заявки
# ---------------------------------------------------------------------------

async def _finish(
    message: Message,
    session: Session,
) -> bool:
    """Сохраняет готовую заявку."""

    draft = session.draft
    user = message.from_user

    # Дополнительный fallback:
    # если клиент не дал контакт, но Telegram username известен,
    # можно использовать его как способ связи.
    if not draft.contact and user and user.username:
        draft.contact = f"@{user.username}"

        log.info(
            "Контакт не получен явно, "
            "использован Telegram username клиента"
        )

    try:
        lead_id = await db.create_lead(
            draft,
            source="bot",
            telegram_id=user.id if user else None,
            telegram_username=(
                user.username
                if user
                else None
            ),
        )

    except Exception:
        log.error(
            "Не удалось сохранить заявку в БД",
            exc_info=True,
        )

        await message.answer(
            "Не получилось сохранить заявку. "
            "Попробуйте, пожалуйста, написать ещё раз через минуту."
        )

        return False

    log.info(
        f"Заявка #{lead_id} сохранена: "
        f"chat_id={message.chat.id}, "
        f"теги={draft.tags or '-'}, "
        f"llm_сбой={session.llm_failed_any}"
    )

    greeting = (
        f"Спасибо, {draft.name}!"
        if draft.name
        else "Спасибо!"
    )

    await message.answer(
        f"{greeting} "
        f"Заявка №{lead_id} принята, "
        "мы свяжемся с вами в ближайшее время."
    )

    _reset(message.chat.id)

    return True


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    _reset(message.chat.id)

    log.info(
        f"/start: chat_id={message.chat.id}"
    )

    await message.answer(GREETING)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message) -> None:
    _reset(message.chat.id)

    log.info(
        f"/cancel: chat_id={message.chat.id}"
    )

    await message.answer(
        "Заявка отменена. "
        "Если захотите начать заново, просто напишите."
    )


def _format_uptime(seconds: float) -> str:
    """Секунды в читаемый вид."""

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
    """Проверка состояния бота."""

    uptime = _format_uptime(
        time.monotonic() - START_MONOTONIC
    )

    log.info(
        f"/status: chat_id={message.chat.id}"
    )

    await message.answer(
        "✅ Бот онлайн\n"
        f"Работает: {uptime}\n"
        f"Запущен: {STARTED_AT:%d.%m.%Y %H:%M:%S}\n"
        f"Диалогов в работе: {len(_sessions)}"
    )


@router.message(Command("id"))
async def cmd_id(message: Message) -> None:
    """Показывает chat_id администратора."""

    await message.answer(
        f"Ваш chat_id: {message.chat.id}"
    )


# ---------------------------------------------------------------------------
# Основной обработчик текста
# ---------------------------------------------------------------------------

@router.message(
    F.text,
    ~F.text.startswith("/"),
)
async def handle_text(
    message: Message,
    llm: OpenRouterClient,
) -> None:
    chat_id = message.chat.id

    # Ограничиваем входной текст.
    text = (message.text or "")[:MAX_INPUT_LEN]

    if not text.strip():
        return

    async with _locks[chat_id]:

        session = _sessions.setdefault(
            chat_id,
            Session(),
        )

        log.info(
            f"Сообщение: "
            f"chat_id={chat_id}, "
            f"длина={len(text)}, "
            f"есть_draft={'да' if session.draft.raw_text else 'нет'}, "
            f"предыдущий_вопрос="
            f"{'да' if session.last_question else 'нет'}"
        )

        try:
            await message.bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.TYPING,
            )

            # ----------------------------------------------------------
            # Один интеллектуальный вызов:
            #
            # LLM получает:
            #   - новое сообщение;
            #   - уже собранную заявку;
            #   - предыдущий вопрос, если он был.
            #
            # Возвращает:
            #   - обновлённый draft;
            #   - один вопрос, если чего-то не хватает.
            # ----------------------------------------------------------

            draft, question = await extract_lead(
                llm=llm,
                text=text,
                draft=session.draft,
                last_question=session.last_question,
            )

            session.draft = draft

            if draft.llm_failed:
                session.llm_failed_any = True

            # ----------------------------------------------------------
            # Окончательное решение принимает Python.
            # Если все обязательные поля есть — сразу сохраняем.
            # ----------------------------------------------------------

            if draft.is_complete:
                await _finish(
                    message,
                    session,
                )
                return

            # ----------------------------------------------------------
            # Если заявка неполная — задаём ровно один вопрос,
            # который сформировала LLM.
            # ----------------------------------------------------------

            if not question:
                # Теоретически extract_lead всегда должен вернуть вопрос
                # при неполной заявке, но оставляем защиту.
                question = (
                    "Подскажите, пожалуйста, "
                    "ещё немного информации для заявки."
                )

            session.last_question = question

            await message.answer(question)

        except Exception:
            log.error(
                f"Ошибка обработки сообщения: "
                f"chat_id={chat_id}",
                exc_info=True,
            )

            await message.answer(
                "Произошла ошибка. "
                "Попробуйте, пожалуйста, ещё раз."
            )


# ---------------------------------------------------------------------------
# Нетекстовые сообщения
# ---------------------------------------------------------------------------

@router.message()
async def handle_other(message: Message) -> None:
    """Пока бот работает только с текстом."""

    log.info(
        f"Нетекстовое сообщение: "
        f"chat_id={message.chat.id}, "
        f"тип={message.content_type}"
    )

    await message.answer(
        "Пока я понимаю только текст. "
        "Напишите, пожалуйста, задачу и контакт сообщением."
    )


# ---------------------------------------------------------------------------
# Уведомления администратору
# ---------------------------------------------------------------------------

def _parse_admin_chat_id(
    raw: Optional[str],
) -> Optional[int]:
    if not raw:
        return None

    try:
        return int(raw.strip())

    except ValueError:
        log.warning(
            "ADMIN_CHAT_ID должен быть числом, "
            "уведомления администратору отключены"
        )

        return None


async def _notify_admin(
    bot: Bot,
    chat_id: Optional[int],
    text: str,
) -> None:
    """Отправляет уведомление администратору."""

    if chat_id is None:
        return

    try:
        await bot.send_message(
            chat_id,
            text,
        )

    except Exception as exc:
        log.warning(
            "Не удалось отправить уведомление "
            f"администратору: {exc}"
        )


# ---------------------------------------------------------------------------
# Меню Telegram
# ---------------------------------------------------------------------------

async def _setup_commands(bot: Bot) -> None:
    """Устанавливает меню команд Telegram."""

    try:
        await bot.set_my_commands(
            [
                BotCommand(
                    command="start",
                    description="Начать заново",
                ),
                BotCommand(
                    command="status",
                    description="Проверить, что бот онлайн",
                ),
                BotCommand(
                    command="cancel",
                    description="Отменить заявку",
                ),
                BotCommand(
                    command="help",
                    description="Помощь",
                ),
            ]
        )

    except Exception as exc:
        log.warning(
            f"Не удалось установить меню команд: {exc}"
        )


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        log.error(
            "Не задан TELEGRAM_BOT_TOKEN "
            "(добавьте его в .env)"
        )

        raise SystemExit(1)

    # Инициализация SQLite.
    await db.init_db()

    # Один LLM-клиент на весь процесс.
    llm = OpenRouterClient()

    bot = Bot(token=token)

    # aiogram автоматически передаст llm в handler
    # благодаря имени параметра.
    dp = Dispatcher(llm=llm)

    dp.include_router(router)

    admin_chat_id = _parse_admin_chat_id(
        os.getenv("ADMIN_CHAT_ID")
    )

    try:
        me = await bot.get_me()

        log.info(
            f"Бот запущен: @{me.username}"
        )

        await _setup_commands(bot)

        await _notify_admin(
            bot,
            admin_chat_id,
            (
                f"🟢 Бот @{me.username} запущен "
                f"({STARTED_AT:%d.%m.%Y %H:%M:%S})"
            ),
        )

        await dp.start_polling(bot)

    finally:
        log.info("Остановка бота")

        await _notify_admin(
            bot,
            admin_chat_id,
            "🔴 Бот остановлен",
        )

        await llm.close()
        await bot.session.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())

    except (KeyboardInterrupt, SystemExit):
        log.info("Бот остановлен")
