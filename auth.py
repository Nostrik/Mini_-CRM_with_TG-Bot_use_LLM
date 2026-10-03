"""Регистрация и авторизация пользователей CRM.

Пользователи лежат в той же SQLite-БД, что и лиды (таблица users). Работа через db._connect().
Пароли хранятся только в виде хеша scrypt (стандартная библиотека, без новых зависимостей).

Модель доступа (открытой саморегистрации нет):
    * Администратор регистрируется сам, но только по секретному ключу ADMIN_API_KEY из окружения
      сервера (register_admin). Ключ знает только владелец системы. Без ключа в окружении
      регистрация отключена совсем. Тем же ключом можно завести нового администратора,
      если доступ к старому потерян.
    * Остальных пользователей создаёт администратор внутри CRM (create_user), задавая им
      временный пароль и роль. Он же может отключать, менять роль, удалять и сбрасывать пароли.

Защита:
    * После MAX_FAILED_ATTEMPTS неверных паролей подряд учётная запись блокируется на LOCK_MINUTES минут.
    * Неверные ключи администратора тоже ограничены: ADMIN_KEY_MAX_FAILURES за ADMIN_KEY_WINDOW_SEC секунд,
      затем регистрация блокируется (счётчик общий для всех посетителей и живёт, пока работает процесс).
    * Сообщение «неверное имя или пароль» одинаково для несуществующего пользователя и неверного пароля.
    * Нельзя удалить, отключить или понизить последнего активного администратора.

Переменные окружения:
    ADMIN_API_KEY - секретный ключ регистрации администратора. Сгенерировать:
                    python -c "import secrets; print(secrets.token_urlsafe(32))"

Порядок: сначала db.init_db_sync(), затем auth.init_auth_sync().
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from typing import Any, Optional

import db
from logger import AppLogger

log = AppLogger()

ROLES = ("admin", "manager")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 128

MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 5

ADMIN_KEY_MAX_FAILURES = 5  # неверных ключей...
ADMIN_KEY_WINDOW_SEC = 300  # ...за это окно (секунды), после чего регистрация блокируется

# Параметры scrypt (OWASP-совместимые для интерактивного входа, ~16 МБ памяти на проверку)
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32

GENERIC_LOGIN_ERROR = "Неверное имя пользователя или пароль"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'manager'
                    CHECK (role IN ({", ".join(f"'{r}'" for r in ROLES)})),
    is_active       INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at   TEXT
);
"""


class AuthError(Exception):
    """Ошибка входа или смены пароля (текст безопасно показывать пользователю)."""


# ---------- пароли ----------
def hash_password(password: str) -> str:
    """Хеш вида scrypt$N$r$p$salt_hex$hash_hex (соль случайная, у каждого пароля своя)."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Проверяет пароль по сохранённому хешу (сравнение за постоянное время)."""
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(hash_hex)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


# хеш-заглушка: проверяем его, когда пользователя нет, чтобы время ответа не выдавало существование имени
_DUMMY_HASH = hash_password("заглушка-для-выравнивания-времени")


# ---------- валидация ----------
def validate_username(username: Optional[str]) -> str:
    """Возвращает имя в нижнем регистре или бросает ValueError."""
    name = (username or "").strip().lower()
    if not USERNAME_RE.match(name):
        raise ValueError("Имя пользователя: 3-32 символа, только латиница, цифры, _ . -")
    return name


def validate_password(password: Optional[str]) -> str:
    password = password or ""
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"Пароль должен быть не короче {MIN_PASSWORD_LEN} символов")
    if len(password) > MAX_PASSWORD_LEN:
        raise ValueError(f"Пароль должен быть не длиннее {MAX_PASSWORD_LEN} символов")
    return password


# ---------- ключ администратора ----------
# Время неудачных попыток ввода ключа (общее для всех посетителей процесса)
_key_failures: deque[float] = deque()
_key_lock = threading.Lock()


def admin_registration_enabled() -> bool:
    """Регистрация администратора включена, только если на сервере задан ADMIN_API_KEY."""
    return bool((os.getenv("ADMIN_API_KEY") or "").strip())


def verify_admin_key(key: Optional[str]) -> None:
    """Проверяет ключ администратора. Бросает AuthError, если ключ неверен, не задан или попыток слишком много."""
    expected = (os.getenv("ADMIN_API_KEY") or "").strip()
    if not expected:
        raise AuthError("Регистрация администратора отключена: на сервере не задан ADMIN_API_KEY")

    with _key_lock:
        now = time.monotonic()
        while _key_failures and now - _key_failures[0] > ADMIN_KEY_WINDOW_SEC:
            _key_failures.popleft()
        if len(_key_failures) >= ADMIN_KEY_MAX_FAILURES:
            log.warning("Регистрация администратора: превышен лимит неверных ключей")
            raise AuthError("Слишком много неверных ключей. Попробуйте позже")

    if not hmac.compare_digest((key or "").strip().encode("utf-8"), expected.encode("utf-8")):
        with _key_lock:
            _key_failures.append(time.monotonic())
        log.warning("Регистрация администратора: неверный ключ")
        raise AuthError("Неверный ключ администратора")

    with _key_lock:
        _key_failures.clear()


# ---------- вспомогательные ----------
def _public(row: sqlite3.Row) -> dict[str, Any]:
    """Данные пользователя без хеша пароля."""
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def _active_admins(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1").fetchone()[0]


def _get_row(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise ValueError("Пользователь не найден")
    return row


def _is_last_active_admin(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    return row["role"] == "admin" and bool(row["is_active"]) and _active_admins(conn) <= 1


# ---------- инициализация ----------
def init_auth_sync() -> None:
    """Создаёт таблицу users. Предупреждает в логе, если войти в CRM будет невозможно."""
    with db._connect() as conn:
        conn.executescript(SCHEMA)
    log.info("Таблица пользователей готова")
    if not admin_registration_enabled() and not has_users():
        log.warning("Нет ни пользователей, ни ADMIN_API_KEY: войти в CRM не получится. Задайте ADMIN_API_KEY в .env")


def has_users() -> bool:
    with db._connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0


# ---------- создание пользователей ----------
def _insert_user(name: str, password: str, role: str) -> int:
    """Добавляет активного пользователя. ValueError, если имя занято."""
    password_hash = hash_password(password)  # тяжёлая операция вне транзакции записи
    try:
        with db._connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role, is_active) VALUES (?, ?, ?, 1)",
                (name, password_hash, role),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("Это имя пользователя уже занято") from None


def register_admin(username: str, password: str, admin_key: Optional[str]) -> dict[str, Any]:
    """Регистрация администратора по ключу ADMIN_API_KEY.

    AuthError: ключ неверен, не задан или превышен лимит попыток. ValueError: плохое имя/пароль или имя занято.
    """
    verify_admin_key(admin_key)
    name = validate_username(username)
    validate_password(password)
    user_id = _insert_user(name, password, "admin")
    log.info(f"Администратор '{name}' зарегистрирован по ключу")
    return get_user(user_id)


def create_user(username: str, password: str, role: str = "manager") -> dict[str, Any]:
    """Создание пользователя администратором внутри CRM (проверка прав выполняется на уровне интерфейса).

    Пароль временный: пользователь может сменить его сам (change_password).
    """
    name = validate_username(username)
    validate_password(password)
    if role not in ROLES:
        raise ValueError(f"Неизвестная роль '{role}', допустимо: {ROLES}")
    user_id = _insert_user(name, password, role)
    log.info(f"Пользователь '{name}' создан администратором: роль={role}")
    return get_user(user_id)


def admin_set_password(user_id: int, new_password: str) -> None:
    """Сброс пароля администратором. Заодно снимает блокировку после неудачных попыток."""
    validate_password(new_password)
    new_hash = hash_password(new_password)
    with db._connect() as conn:
        _get_row(conn, user_id)  # ValueError, если пользователя нет
        conn.execute(
            "UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
            (new_hash, user_id),
        )
    log.info(f"Пароль пользователя #{user_id} сброшен администратором")


# ---------- вход ----------
def authenticate(username: str, password: str) -> dict[str, Any]:
    """Проверяет логин и пароль. Возвращает данные пользователя или бросает AuthError."""
    name = (username or "").strip().lower()
    password = password or ""

    with db._connect() as conn:
        row = conn.execute(
            """SELECT *, (locked_until IS NOT NULL AND locked_until > datetime('now')) AS locked
               FROM users WHERE username = ?""",
            (name,),
        ).fetchone()

    if row is None:
        verify_password(password, _DUMMY_HASH)
        log.warning(f"Вход: неизвестное имя '{name[:32]}'")
        raise AuthError(GENERIC_LOGIN_ERROR)

    if row["locked"]:
        log.warning(f"Вход: '{name}' временно заблокирован из-за неудачных попыток")
        raise AuthError(f"Слишком много неудачных попыток. Попробуйте через {LOCK_MINUTES} мин.")

    if not verify_password(password, row["password_hash"]):
        _register_failure(row["id"])
        log.warning(f"Вход: неверный пароль для '{name}'")
        raise AuthError(GENERIC_LOGIN_ERROR)

    if not row["is_active"]:
        log.info(f"Вход: '{name}' ещё не активирован или отключён")
        raise AuthError("Учётная запись отключена администратором")

    _register_success(row["id"])
    log.info(f"Вход выполнен: '{name}'")
    return get_user(row["id"])


def _register_failure(user_id: int) -> None:
    """Считает неудачную попытку; на MAX_FAILED_ATTEMPTS-й блокирует вход и обнуляет счётчик."""
    with db._connect() as conn:
        conn.execute(
            """UPDATE users SET
                   locked_until = CASE WHEN failed_attempts + 1 >= ?
                                       THEN datetime('now', ?) ELSE locked_until END,
                   failed_attempts = CASE WHEN failed_attempts + 1 >= ?
                                          THEN 0 ELSE failed_attempts + 1 END
               WHERE id = ?""",
            (MAX_FAILED_ATTEMPTS, f"+{LOCK_MINUTES} minutes", MAX_FAILED_ATTEMPTS, user_id),
        )


def _register_success(user_id: int) -> None:
    with db._connect() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,),
        )


# ---------- чтение ----------
def get_user(user_id: int) -> Optional[dict[str, Any]]:
    with db._connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public(row) if row else None


def list_users() -> list[dict[str, Any]]:
    with db._connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY created_at, id").fetchall()
    return [_public(r) for r in rows]


# ---------- управление пользователями (для администратора) ----------
def set_user_active(user_id: int, active: bool, actor_id: Optional[int] = None) -> None:
    """Активирует или отключает пользователя. Нельзя отключить себя (actor_id) и последнего администратора."""
    if not active and actor_id is not None and actor_id == user_id:
        raise ValueError("Нельзя отключить самого себя")
    with db._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _get_row(conn, user_id)
        if not active and _is_last_active_admin(conn, row):
            raise ValueError("Нельзя отключить последнего активного администратора")
        conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, user_id))
    log.info(f"Пользователь #{user_id}: активен={active}")


def set_user_role(user_id: int, role: str, actor_id: Optional[int] = None) -> None:
    """Меняет роль. Нельзя менять собственную роль (actor_id) и понизить последнего администратора."""
    if role not in ROLES:
        raise ValueError(f"Неизвестная роль '{role}', допустимо: {ROLES}")
    if actor_id is not None and actor_id == user_id:
        raise ValueError("Нельзя изменить собственную роль")
    with db._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _get_row(conn, user_id)
        if role != "admin" and _is_last_active_admin(conn, row):
            raise ValueError("Нельзя понизить последнего активного администратора")
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    log.info(f"Пользователь #{user_id}: роль={role}")


def delete_user(user_id: int) -> None:
    """Удаляет пользователя. Нельзя удалить последнего администратора."""
    with db._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _get_row(conn, user_id)
        if _is_last_active_admin(conn, row):
            raise ValueError("Нельзя удалить последнего активного администратора")
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    log.info(f"Пользователь #{user_id} удалён")


def change_password(user_id: int, old_password: str, new_password: str) -> None:
    """Смена собственного пароля. AuthError, если текущий пароль неверен; ValueError, если новый слабый."""
    validate_password(new_password)
    with db._connect() as conn:
        row = _get_row(conn, user_id)
    if not verify_password(old_password or "", row["password_hash"]):
        log.warning(f"Смена пароля: неверный текущий пароль у пользователя #{user_id}")
        raise AuthError("Текущий пароль неверен")
    new_hash = hash_password(new_password)
    with db._connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    log.info(f"Пользователь #{user_id} сменил пароль")
