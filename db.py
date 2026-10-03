"""SQLite-хранилище мини-CRM: лиды, теги, связь лид-тег.

Устройство:
    * Ядро синхронное (стандартный sqlite3, без лишних зависимостей): его напрямую
      использует Streamlit-интерфейс (list_leads, create_lead_sync, set_lead_tags ...).
    * Для бота есть асинхронные обёртки init_db() и create_lead(): они гоняют
      синхронный код в отдельном потоке (asyncio.to_thread), чтобы не блокировать event loop.
    * Бот и Streamlit - разные процессы с одним файлом БД. Включён режим WAL и busy_timeout,
      поэтому параллельные чтение и запись безопасны.

Переменные окружения:
    DB_PATH - путь к файлу БД (по умолчанию crm.db). На хостинге укажите путь на постоянном
              томе (volume), иначе файл пропадёт при редеплое.

Время в БД хранится в UTC (CURRENT_TIMESTAMP).
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from dotenv import load_dotenv

from logger import AppLogger

if TYPE_CHECKING:  # только для подсказок типов, чтобы db.py не тянул extract и openrouter_api
    from extract import LeadDraft

load_dotenv()

log = AppLogger()

DB_PATH = os.getenv("DB_PATH", "crm.db")

SOURCES = ("bot", "manual", "tg_account")
STATUSES = ("new", "in_progress", "done", "rejected")
EDITABLE_FIELDS = ("name", "contact", "request", "status")

MAX_TAG_LEN = 40
MAX_LIMIT = 900  # ограничение на число строк за запрос (и на число параметров IN)

_LIKE_ESCAPE = " ESCAPE '\\'"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS leads (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT,
    contact           TEXT,
    request           TEXT,
    source            TEXT NOT NULL DEFAULT 'manual'
                      CHECK (source IN ({", ".join(f"'{s}'" for s in SOURCES)})),
    status            TEXT NOT NULL DEFAULT 'new'
                      CHECK (status IN ({", ".join(f"'{s}'" for s in STATUSES)})),
    raw_text          TEXT,
    telegram_id       INTEGER,
    telegram_username TEXT,
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tags (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS lead_tags (
    lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
    PRIMARY KEY (lead_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_leads_created  ON leads(created_at);
CREATE INDEX IF NOT EXISTS idx_leads_telegram ON leads(telegram_id);
CREATE INDEX IF NOT EXISTS idx_lead_tags_tag  ON lead_tags(tag_id);
"""


# ---------- подключение ----------
@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Соединение на одну операцию: commit при успехе, rollback при ошибке, close всегда."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # в SQLite внешние ключи по умолчанию выключены
    conn.execute("PRAGMA busy_timeout = 10000")
    # LIKE и lower() в SQLite понимают регистр только для ASCII, а кириллицу нет:
    # подставляем свою функцию, чтобы поиск «ирина» находил «Ирина».
    conn.create_function("py_lower", 1, lambda s: s.lower() if isinstance(s, str) else s)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db_sync(seed_tags: Optional[list[str]] = None) -> None:
    """Создаёт таблицы (если их нет). seed_tags: теги, которые нужно завести заранее."""
    Path(DB_PATH).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # бот и Streamlit работают с файлом параллельно
        conn.executescript(SCHEMA)
        for tag in seed_tags or []:
            _get_or_create_tag(conn, tag)
    log.info(f"БД инициализирована: {DB_PATH}")


# ---------- вспомогательные ----------
def normalize_tag(name: Optional[str]) -> str:
    """Единый вид тега: без лишних пробелов, в нижнем регистре, не длиннее MAX_TAG_LEN."""
    return " ".join((name or "").split()).lower()[:MAX_TAG_LEN]


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _get_or_create_tag(conn: sqlite3.Connection, name: str) -> Optional[int]:
    tag = normalize_tag(name)
    if not tag:
        return None
    conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
    return conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()["id"]


def _add_tags(conn: sqlite3.Connection, lead_id: int, tags: list[str]) -> None:
    for tag in tags:
        tag_id = _get_or_create_tag(conn, tag)
        if tag_id is not None:
            conn.execute(
                "INSERT OR IGNORE INTO lead_tags (lead_id, tag_id) VALUES (?, ?)", (lead_id, tag_id)
            )


def _attach_tags(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    """Добавляет в каждую строку ключ 'tags' со списком тегов лида."""
    for row in rows:
        row["tags"] = []
    if not rows:
        return
    by_id = {row["id"]: row for row in rows}
    placeholders = ",".join("?" * len(by_id))
    cursor = conn.execute(
        f"""SELECT lt.lead_id, t.name
            FROM lead_tags lt JOIN tags t ON t.id = lt.tag_id
            WHERE lt.lead_id IN ({placeholders})
            ORDER BY t.name""",
        list(by_id),
    )
    for rec in cursor:
        by_id[rec["lead_id"]]["tags"].append(rec["name"])


# ---------- лиды: создание ----------
def create_lead_sync(
    *,
    name: Optional[str] = None,
    contact: Optional[str] = None,
    request: Optional[str] = None,
    source: str = "manual",
    status: str = "new",
    tags: Optional[list[str]] = None,
    raw_text: Optional[str] = None,
    telegram_id: Optional[int] = None,
    telegram_username: Optional[str] = None,
) -> int:
    """Создаёт лид вместе с тегами и возвращает его id.

    Нужно заполнить хотя бы одно из полей: name, contact, request, raw_text.
    """
    if source not in SOURCES:
        raise ValueError(f"Неизвестный источник '{source}', допустимо: {SOURCES}")
    if status not in STATUSES:
        raise ValueError(f"Неизвестный статус '{status}', допустимо: {STATUSES}")

    name, contact, request, raw_text = (_none_if_blank(v) for v in (name, contact, request, raw_text))
    if not any((name, contact, request, raw_text)):
        raise ValueError("Укажите хотя бы имя, контакт или запрос")

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO leads
                   (name, contact, request, source, status, raw_text, telegram_id, telegram_username)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, contact, request, source, status, raw_text, telegram_id, _none_if_blank(telegram_username)),
        )
        lead_id = cur.lastrowid
        _add_tags(conn, lead_id, tags or [])

    log.info(f"Лид #{lead_id} создан: источник={source}, тегов={len(tags or [])}")
    return lead_id


# ---------- лиды: чтение ----------
def get_lead(lead_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
        if row is None:
            return None
        lead = dict(row)
        _attach_tags(conn, [lead])
    return lead


def list_leads(
    tag: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    only_untagged: bool = False,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Список лидов, новые сверху. Каждый лид содержит ключ 'tags' (список).

    tag           - показать только лидов с этим тегом
    only_untagged - показать только лидов без тегов
    search        - поиск по имени, контакту, запросу и тексту переписки (без учёта регистра)
    """
    where: list[str] = []
    params: list[Any] = []

    if tag:
        # подзапрос, а не JOIN, чтобы у найденных лидов в 'tags' остались все теги, а не только искомый
        where.append(
            "l.id IN (SELECT lt.lead_id FROM lead_tags lt "
            "JOIN tags t ON t.id = lt.tag_id WHERE t.name = ?)"
        )
        params.append(normalize_tag(tag))
    if only_untagged:
        where.append("NOT EXISTS (SELECT 1 FROM lead_tags lt WHERE lt.lead_id = l.id)")
    if source:
        where.append("l.source = ?")
        params.append(source)
    if status:
        where.append("l.status = ?")
        params.append(status)
    if search and search.strip():
        pattern = "%" + _escape_like(search.strip().lower()) + "%"
        columns = ("name", "contact", "request", "raw_text")
        where.append("(" + " OR ".join(f"py_lower(l.{c}) LIKE ?{_LIKE_ESCAPE}" for c in columns) + ")")
        params.extend([pattern] * len(columns))

    sql = "SELECT l.* FROM leads l"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY l.created_at DESC, l.id DESC LIMIT ?"
    params.append(max(1, min(int(limit), MAX_LIMIT)))

    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
        _attach_tags(conn, rows)
    return rows


# ---------- лиды: изменение ----------
def update_lead(lead_id: int, **fields: Any) -> bool:
    """Меняет name / contact / request / status. Возвращает False, если лида нет."""
    unknown = set(fields) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"Эти поля менять нельзя: {sorted(unknown)}")
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"Неизвестный статус '{fields['status']}', допустимо: {STATUSES}")
    if not fields:
        return False

    cleaned = {k: (v if k == "status" else _none_if_blank(v)) for k, v in fields.items()}
    assignments = ", ".join(f"{col} = ?" for col in cleaned)  # имена колонок только из белого списка
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE leads SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (*cleaned.values(), lead_id),
        )
        updated = cur.rowcount > 0
    if updated:
        log.info(f"Лид #{lead_id} обновлён: поля={list(cleaned)}")
    return updated


def set_lead_tags(lead_id: int, tags: list[str]) -> bool:
    """Полностью заменяет теги лида. Возвращает False, если лида нет."""
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM leads WHERE id = ?", (lead_id,)).fetchone() is None:
            return False
        conn.execute("DELETE FROM lead_tags WHERE lead_id = ?", (lead_id,))
        _add_tags(conn, lead_id, tags)
        conn.execute("UPDATE leads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (lead_id,))
    log.info(f"Теги лида #{lead_id} заменены: {len(tags)} шт.")
    return True


def add_lead_tags(lead_id: int, tags: list[str]) -> bool:
    """Добавляет теги к уже существующим. Возвращает False, если лида нет."""
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM leads WHERE id = ?", (lead_id,)).fetchone() is None:
            return False
        _add_tags(conn, lead_id, tags)
        conn.execute("UPDATE leads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (lead_id,))
    log.info(f"К лиду #{lead_id} добавлены теги: {len(tags)} шт.")
    return True


def delete_lead(lead_id: int) -> bool:
    """Удаляет лид (связи с тегами удаляются каскадом)."""
    with _connect() as conn:
        deleted = conn.execute("DELETE FROM leads WHERE id = ?", (lead_id,)).rowcount > 0
    if deleted:
        log.info(f"Лид #{lead_id} удалён")
    return deleted


# ---------- теги ----------
def list_tags() -> list[dict[str, Any]]:
    """Все теги с числом лидов: [{'name': 'сайт', 'count': 3}, ...] (популярные первыми)."""
    with _connect() as conn:
        cursor = conn.execute(
            """SELECT t.name AS name, COUNT(lt.lead_id) AS count
               FROM tags t LEFT JOIN lead_tags lt ON lt.tag_id = t.id
               GROUP BY t.id
               ORDER BY count DESC, t.name"""
        )
        return [dict(r) for r in cursor]


# ---------- асинхронные обёртки для бота ----------
async def init_db(seed_tags: Optional[list[str]] = None) -> None:
    await asyncio.to_thread(init_db_sync, seed_tags)


async def create_lead(
    draft: "LeadDraft",
    source: str,
    telegram_id: Optional[int] = None,
    telegram_username: Optional[str] = None,
) -> int:
    """Сохраняет черновик заявки (LeadDraft из extract.py) как лид. Возвращает id лида."""
    try:
        return await asyncio.to_thread(
            create_lead_sync,
            name=draft.name,
            contact=draft.contact,
            request=draft.request,
            source=source,
            tags=list(draft.tags),
            raw_text=draft.raw_text,
            telegram_id=telegram_id,
            telegram_username=telegram_username,
        )
    except Exception:
        log.error("Ошибка сохранения лида в БД", exc_info=True)
        raise
    