"""Мини-CRM на Streamlit: таблица лидов, фильтры, ручное добавление, правка тегов и статусов.

Запуск:  streamlit run app.py   (или: poetry run streamlit run app.py)

Переменные окружения (все необязательные):
    DB_PATH       - путь к файлу SQLite (общий с ботом), по умолчанию crm.db
    ADMIN_API_KEY - секретный ключ, по которому регистрируется администратор. Без него регистрация
                    отключена. Остальных пользователей администратор добавляет внутри CRM.
    APP_TZ        - часовой пояс для отображения времени, по умолчанию Europe/Moscow
                    (на Windows для названий поясов нужен пакет tzdata: poetry add tzdata)

Бот (bot2.py) и этот интерфейс - отдельные процессы, общаются только через файл БД.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

# set_page_config должен быть первой командой Streamlit в скрипте
st.set_page_config(page_title="Мини-CRM", page_icon="📋", layout="wide")

import auth  # noqa: E402
import db  # noqa: E402
from extract2 import ALLOWED_TAGS  # noqa: E402
from logger import AppLogger  # noqa: E402

SOURCE_LABELS = {"bot": "Telegram-бот", "manual": "Вручную", "tg_account": "Telegram-аккаунт"}
STATUS_LABELS = {"new": "Новый", "in_progress": "В работе", "done": "Закрыт", "rejected": "Отказ"}
ROLE_LABELS = {"admin": "Администратор", "manager": "Менеджер"}
SESSION_USER_KEY = "user"  # ключ в st.session_state: {"id", "username", "role"}

TABLE_COLUMNS = ["ID", "Создан", "Имя", "Контакт", "Запрос", "Теги", "Источник", "Статус"]
UNTAGGED = "__untagged__"  # служебное значение фильтра «Без тега»


# ---------- инфраструктура ----------
@st.cache_resource
def get_logger() -> AppLogger:
    """Один логгер на процесс (скрипт Streamlit перезапускается при каждом клике)."""
    return AppLogger()


@st.cache_resource
def init_storage() -> bool:
    """Создаёт таблицы и заводит стандартные теги один раз за процесс."""
    db.init_db_sync(seed_tags=ALLOWED_TAGS)
    auth.init_auth_sync()  # после db: использует то же соединение и файл
    return True


def _load_tz() -> tzinfo:
    name = os.getenv("APP_TZ", "Europe/Moscow")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc  # нет базы часовых поясов (часто на Windows): показываем UTC


TZ = _load_tz()
log = get_logger()


def _fmt_ts(value: Optional[str]) -> str:
    """Время из БД (UTC) в локальный часовой пояс."""
    if not value:
        return ""
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.astimezone(TZ).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return value


def flash(message: str, icon: str = "✅") -> None:
    """Сообщение, которое покажется после ближайшего st.rerun()."""
    st.session_state["flash"] = (message, icon)


def tag_options(extra: Optional[list[str]] = None) -> list[str]:
    """Все существующие теги (+ теги конкретного лида, на случай если их нет в списке)."""
    names = [t["name"] for t in db.list_tags()]
    for tag in extra or []:
        if tag not in names:
            names.append(tag)
    return names


# ---------- регистрация и вход ----------
def _start_session(user: dict) -> None:
    st.session_state[SESSION_USER_KEY] = {"id": user["id"], "username": user["username"], "role": user["role"]}


def _end_session() -> None:
    st.session_state.pop(SESSION_USER_KEY, None)


def _render_login_form() -> None:
    with st.form("login_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти", type="primary")
    if not submitted:
        return

    user = None
    try:
        user = auth.authenticate(username, password)
    except auth.AuthError as e:
        st.error(str(e))
    except Exception:
        log.error("Ошибка при входе", exc_info=True)
        st.error("Не удалось выполнить вход, попробуйте ещё раз")

    if user is not None:
        _start_session(user)
        st.rerun()


def _render_register_form() -> None:
    if not auth.admin_registration_enabled():
        st.warning("Регистрация администратора отключена: на сервере не задан ADMIN_API_KEY.")
        return

    st.caption(
        "Регистрация только для администратора: нужен ключ. "
        "Остальных пользователей добавляет администратор внутри CRM."
    )

    with st.form("register_form"):
        admin_key = st.text_input("Ключ администратора (API key)", type="password")
        username = st.text_input("Имя пользователя", help="3-32 символа: латиница, цифры, _ . -")
        password = st.text_input("Пароль", type="password", help=f"Не короче {auth.MIN_PASSWORD_LEN} символов")
        password2 = st.text_input("Повторите пароль", type="password")
        submitted = st.form_submit_button("Зарегистрироваться", type="primary")
    if not submitted:
        return

    if password != password2:
        st.error("Пароли не совпадают")
        return

    user = None
    try:
        user = auth.register_admin(username, password, admin_key)
    except (auth.AuthError, ValueError) as e:
        st.error(str(e))
    except Exception:
        log.error("Ошибка при регистрации администратора", exc_info=True)
        st.error("Не удалось зарегистрироваться, попробуйте ещё раз")

    if user is not None:
        _start_session(user)
        flash(f"Добро пожаловать, {user['username']}!", icon="👋")
        st.rerun()


AUTH_FORM_WIDTH_PX = 450


def _limit_auth_width() -> None:
    """Сужает страницу входа/регистрации до AUTH_FORM_WIDTH_PX и центрирует.
    Вызывается только на экране входа, поэтому сама CRM после входа остаётся широкой."""
    st.markdown(
        f"""
        <style>
        .stMainBlockContainer, .block-container {{
            max-width: {AUTH_FORM_WIDTH_PX}px;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth_gate() -> Optional[dict]:
    """Возвращает текущего пользователя. Если не вошёл, рисует вход/регистрацию и возвращает None."""
    session_user = st.session_state.get(SESSION_USER_KEY)
    if session_user:
        # каждый раз сверяемся с БД: администратор мог отключить или удалить пользователя
        fresh = auth.get_user(session_user["id"])
        if fresh and fresh["is_active"]:
            _start_session(fresh)  # заодно обновит роль, если её изменили
            return st.session_state[SESSION_USER_KEY]
        _end_session()
        st.warning("Доступ закрыт администратором или учётная запись удалена.")

    _limit_auth_width()
    st.title("📋 Мини-CRM")
    tab_login, tab_register = st.tabs(["Вход", "Регистрация администратора"])
    with tab_login:
        _render_login_form()
    with tab_register:
        _render_register_form()
    return None


def render_account_sidebar(user: dict) -> None:
    """Блок «кто вошёл» в сайдбаре: выход и смена пароля."""
    st.sidebar.markdown(f"👤 **{user['username']}** · {ROLE_LABELS.get(user['role'], user['role'])}")
    if st.sidebar.button("Выйти"):
        log.info(f"Выход: '{user['username']}'")
        _end_session()
        st.rerun()

    with st.sidebar.expander("Сменить пароль"):
        with st.form("change_password_form", clear_on_submit=True):
            old = st.text_input("Текущий пароль", type="password")
            new = st.text_input("Новый пароль", type="password")
            new2 = st.text_input("Повторите новый пароль", type="password")
            submitted = st.form_submit_button("Сменить пароль")
        if submitted:
            if new != new2:
                st.error("Новые пароли не совпадают")
            else:
                try:
                    auth.change_password(user["id"], old, new)
                except (auth.AuthError, ValueError) as e:
                    st.error(str(e))
                except Exception:
                    log.error("Ошибка смены пароля", exc_info=True)
                    st.error("Не удалось сменить пароль, попробуйте ещё раз")
                else:
                    st.success("Пароль изменён")


# ---------- данные для таблицы ----------
def build_dataframe(leads: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "ID": lead["id"],
            "Создан": _fmt_ts(lead["created_at"]),
            "Имя": lead["name"] or "",
            "Контакт": lead["contact"] or "",
            "Запрос": lead["request"] or "",
            "Теги": ", ".join(lead["tags"]),
            "Источник": SOURCE_LABELS.get(lead["source"], lead["source"]),
            "Статус": STATUS_LABELS.get(lead["status"], lead["status"]),
        }
        for lead in leads
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


# ---------- блоки интерфейса ----------
def render_metrics() -> None:
    all_leads = db.list_leads(limit=db.MAX_LIMIT)
    cols = st.columns(5)
    cols[0].metric("Всего лидов", len(all_leads))
    cols[1].metric("Новых", sum(1 for l in all_leads if l["status"] == "new"))
    cols[2].metric("Из бота", sum(1 for l in all_leads if l["source"] == "bot"))
    cols[3].metric("Вручную", sum(1 for l in all_leads if l["source"] == "manual"))
    cols[4].metric("Без тегов", sum(1 for l in all_leads if not l["tags"]))


def render_sidebar_filters() -> dict:
    """Рисует фильтры в сайдбаре и возвращает параметры для db.list_leads()."""
    st.sidebar.header("Фильтры")
    if st.sidebar.button("🔄 Обновить данные"):
        st.rerun()

    search = st.sidebar.text_input("Поиск", placeholder="имя, контакт, запрос, переписка")

    tag_choices: dict[str, Optional[str]] = {"Все теги": None, "Без тега": UNTAGGED}
    for t in db.list_tags():
        tag_choices[f"{t['name']} ({t['count']})"] = t["name"]
    tag_label = st.sidebar.selectbox("Тег", options=list(tag_choices))
    tag_value = tag_choices[tag_label]

    source = st.sidebar.selectbox(
        "Источник",
        options=[None, *db.SOURCES],
        format_func=lambda v: "Все источники" if v is None else SOURCE_LABELS[v],
    )
    status = st.sidebar.selectbox(
        "Статус",
        options=[None, *db.STATUSES],
        format_func=lambda v: "Все статусы" if v is None else STATUS_LABELS[v],
    )

    return {
        "tag": tag_value if tag_value not in (None, UNTAGGED) else None,
        "only_untagged": tag_value == UNTAGGED,
        "source": source,
        "status": status,
        "search": search,
        "limit": db.MAX_LIMIT,
    }


def render_add_form() -> None:
    st.subheader("Новый лид вручную")
    with st.form("add_lead_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Имя")
        contact = c2.text_input("Контакт", placeholder="телефон, email или @username")
        request = st.text_area("Запрос", placeholder="Что нужно клиенту")
        tags = st.multiselect(
            "Теги",
            options=tag_options(),
            accept_new_options=True,
            help="Выберите из списка или введите новый тег и нажмите Enter",
        )
        submitted = st.form_submit_button("Добавить лид", type="primary")

    if not submitted:
        return

    created_id = None
    try:
        created_id = db.create_lead_sync(
            name=name, contact=contact, request=request, source="manual", tags=tags
        )
    except ValueError as e:
        st.error(str(e))
    except Exception:
        log.error("Не удалось создать лид вручную", exc_info=True)
        st.error("Не удалось сохранить лид, попробуйте ещё раз")

    if created_id is not None:
        flash(f"Лид #{created_id} добавлен")
        st.rerun()


def render_lead_editor(lead_id: int) -> None:
    lead = db.get_lead(lead_id)
    if lead is None:
        st.warning("Лид не найден (возможно, его уже удалили)")
        return

    st.subheader(f"Лид #{lead['id']}")
    meta = [
        f"Источник: **{SOURCE_LABELS.get(lead['source'], lead['source'])}**",
        f"Создан: {_fmt_ts(lead['created_at'])}",
    ]
    if lead["telegram_username"]:
        username = lead["telegram_username"]
        meta.append(f"Telegram: [@{username}](https://t.me/{username})")
    st.markdown(" · ".join(meta))

    if lead["raw_text"]:
        with st.expander("Исходное сообщение клиента"):
            st.text(lead["raw_text"])

    statuses = list(db.STATUSES)
    with st.form(f"edit_form_{lead_id}"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Имя", value=lead["name"] or "", key=f"name_{lead_id}")
        contact = c2.text_input("Контакт", value=lead["contact"] or "", key=f"contact_{lead_id}")
        request = st.text_area("Запрос", value=lead["request"] or "", key=f"request_{lead_id}")
        c3, c4 = st.columns(2)
        status = c3.selectbox(
            "Статус",
            options=statuses,
            index=statuses.index(lead["status"]),
            format_func=lambda s: STATUS_LABELS[s],
            key=f"status_{lead_id}",
        )
        tags = c4.multiselect(
            "Теги",
            options=tag_options(lead["tags"]),
            default=lead["tags"],
            accept_new_options=True,
            key=f"tags_{lead_id}",
        )
        saved = st.form_submit_button("💾 Сохранить", type="primary")

    if saved:
        ok = False
        try:
            db.update_lead(lead_id, name=name, contact=contact, request=request, status=status)
            db.set_lead_tags(lead_id, tags)
            ok = True
        except ValueError as e:
            st.error(str(e))
        except Exception:
            log.error(f"Не удалось сохранить лид #{lead_id}", exc_info=True)
            st.error("Не удалось сохранить изменения, попробуйте ещё раз")
        if ok:
            flash(f"Лид #{lead_id} сохранён")
            st.rerun()

    with st.expander("Удалить лид"):
        confirm = st.checkbox("Да, удалить безвозвратно", key=f"confirm_del_{lead_id}")
        if st.button("🗑️ Удалить", key=f"del_{lead_id}", disabled=not confirm):
            deleted = False
            try:
                deleted = db.delete_lead(lead_id)
            except Exception:
                log.error(f"Не удалось удалить лид #{lead_id}", exc_info=True)
                st.error("Не удалось удалить лид, попробуйте ещё раз")
            if deleted:
                # новая версия ключа таблицы сбрасывает выделение строки
                st.session_state["table_version"] = st.session_state.get("table_version", 0) + 1
                flash(f"Лид #{lead_id} удалён", icon="🗑️")
                st.rerun()


def render_leads_tab(leads: list[dict]) -> None:
    if not leads:
        st.info("Лидов по выбранным фильтрам нет. Измените фильтры или добавьте лида вручную.")
        return

    df = build_dataframe(leads)
    top_left, top_right = st.columns([3, 1])
    top_left.caption(f"Найдено: {len(df)}. Выберите строку, чтобы открыть карточку лида.")
    top_right.download_button(
        "⬇️ Скачать CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),  # utf-8-sig: чтобы Excel открыл кириллицу
        file_name="leads.csv",
        mime="text/csv",
    )

    version = st.session_state.get("table_version", 0)
    event = st.dataframe(
        df,
        key=f"leads_table_{version}",
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        width="stretch",
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Создан": st.column_config.TextColumn("Создан", width="small"),
            "Запрос": st.column_config.TextColumn("Запрос", width="large"),
        },
    )

    selected = event.selection.rows
    if selected and selected[0] < len(df):
        st.divider()
        render_lead_editor(int(df.iloc[selected[0]]["ID"]))


def render_tags_tab() -> None:
    stats = db.list_tags()
    if not stats:
        st.info("Тегов пока нет.")
        return
    df = pd.DataFrame(stats).rename(columns={"name": "Тег", "count": "Лидов"})
    st.caption("Теги создаются при добавлении лида и в карточке лида. Чтобы увидеть лидов по тегу, выберите его в фильтре слева.")
    left, right = st.columns([1, 2])
    left.dataframe(df, hide_index=True, width="stretch")
    right.bar_chart(df, x="Тег", y="Лидов")


def _render_add_user_form() -> None:
    """Администратор заводит сотрудника: имя, временный пароль и роль."""
    roles = list(auth.ROLES)
    with st.expander("➕ Добавить пользователя"):
        with st.form("add_user_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            username = c1.text_input("Имя пользователя", help="3-32 символа: латиница, цифры, _ . -")
            password = c2.text_input(
                "Временный пароль",
                type="password",
                help=f"Не короче {auth.MIN_PASSWORD_LEN} символов. Сообщите его сотруднику: он сможет сменить пароль сам",
            )
            role = st.selectbox(
                "Роль", options=roles, index=roles.index("manager"), format_func=lambda r: ROLE_LABELS[r]
            )
            submitted = st.form_submit_button("Создать пользователя", type="primary")

        if not submitted:
            return

        created = None
        try:
            created = auth.create_user(username, password, role)
        except ValueError as e:
            st.error(str(e))
        except Exception:
            log.error("Не удалось создать пользователя", exc_info=True)
            st.error("Не удалось создать пользователя, попробуйте ещё раз")

        if created is not None:
            flash(f"Пользователь «{created['username']}» создан", icon="👤")
            st.rerun()


def _render_reset_password_form(users: list[dict]) -> None:
    """Сброс пароля сотруднику (например, если он его забыл или заблокирован после неудачных попыток)."""
    names = {u["id"]: u["username"] for u in users}
    with st.expander("🔑 Сбросить пароль пользователю"):
        with st.form("reset_password_form", clear_on_submit=True):
            target_id = st.selectbox("Пользователь", options=list(names), format_func=lambda uid: names[uid])
            new_password = st.text_input("Новый пароль", type="password")
            submitted = st.form_submit_button("Сбросить пароль")

        if not submitted:
            return
        try:
            auth.admin_set_password(target_id, new_password)
        except ValueError as e:
            st.error(str(e))
        except Exception:
            log.error(f"Не удалось сбросить пароль пользователю #{target_id}", exc_info=True)
            st.error("Не удалось сбросить пароль, попробуйте ещё раз")
        else:
            st.success(f"Пароль для «{names[target_id]}» обновлён. Сообщите его пользователю.")


def render_users_tab(current_user: dict) -> None:
    """Управление пользователями (только для администратора)."""
    st.subheader("Пользователи")
    st.caption("Добавляйте сотрудников, меняйте роли, отключайте доступ и сбрасывайте пароли.")

    users = auth.list_users()
    _render_add_user_form()

    for u in users:
        is_me = u["id"] == current_user["id"]
        cols = st.columns([3, 2, 2, 2, 2, 1])
        cols[0].markdown(f"**{u['username']}**{' (вы)' if is_me else ''}  \n:gray[создан {_fmt_ts(u['created_at'])}]")
        cols[1].write(ROLE_LABELS.get(u["role"], u["role"]))
        cols[2].write("✅ Активен" if u["is_active"] else "⛔ Отключён")

        action = None
        if cols[3].button(
            "Отключить" if u["is_active"] else "Активировать",
            key=f"user_active_{u['id']}",
            disabled=is_me,
            help="Нельзя отключить самого себя" if is_me else None,
        ):
            action = ("active", not u["is_active"])
        if cols[4].button(
            "Сделать менеджером" if u["role"] == "admin" else "Сделать админом",
            key=f"user_role_{u['id']}",
            disabled=is_me,
            help="Нельзя изменить собственную роль" if is_me else None,
        ):
            action = ("role", "manager" if u["role"] == "admin" else "admin")
        if cols[5].button("🗑️", key=f"user_del_{u['id']}", disabled=is_me, help="Удалить пользователя"):
            action = ("delete", None)

        if action is None:
            continue

        ok = False
        try:
            kind, value = action
            if kind == "active":
                auth.set_user_active(u["id"], value, actor_id=current_user["id"])
            elif kind == "role":
                auth.set_user_role(u["id"], value, actor_id=current_user["id"])
            else:
                auth.delete_user(u["id"])
            ok = True
        except ValueError as e:
            st.error(str(e))
        except Exception:
            log.error(f"Ошибка управления пользователем #{u['id']}", exc_info=True)
            st.error("Не удалось выполнить действие, попробуйте ещё раз")
        if ok:
            flash(f"Изменения для «{u['username']}» сохранены")
            st.rerun()

    st.divider()
    _render_reset_password_form(users)


# ---------- точка входа ----------
def main() -> None:
    init_storage()  # таблицы нужны до проверки входа

    user = render_auth_gate()
    if user is None:
        st.stop()

    st.title("📋 Мини-CRM")

    flash_data = st.session_state.pop("flash", None)
    if flash_data:
        message, icon = flash_data
        st.toast(message, icon=icon)

    render_account_sidebar(user)
    filters = render_sidebar_filters()
    render_metrics()

    leads = db.list_leads(**filters)

    is_admin = user["role"] == "admin"
    tab_names = ["📋 Лиды", "➕ Добавить лида", "🏷️ Теги"]
    if is_admin:
        tab_names.append("👥 Пользователи")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_leads_tab(leads)
    with tabs[1]:
        render_add_form()
    with tabs[2]:
        render_tags_tab()
    if is_admin:
        with tabs[3]:
            render_users_tab(user)


main()
