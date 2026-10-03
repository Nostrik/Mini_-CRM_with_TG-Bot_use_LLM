"""Тесты db.py. Каждый тест работает с отдельным временным файлом БД (tmp_path)."""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

import db


# ---------- фикстуры и помощники ----------
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Подменяет путь к БД на временный файл и создаёт схему."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    db.init_db_sync()


def make_lead(**kwargs) -> int:
    params = {"name": "Ирина", "contact": "@irina_coffee", "request": "Сайт для кофейни"}
    params.update(kwargs)
    return db.create_lead_sync(**params)


def draft(**kwargs):
    """Заглушка LeadDraft (db.py не обязан импортировать extract)."""
    data = {"name": None, "contact": None, "request": None, "tags": [], "raw_text": ""}
    data.update(kwargs)
    return SimpleNamespace(**data)


# ---------- инициализация ----------
def test_init_creates_tables():
    with db._connect() as conn:
        names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"leads", "tags", "lead_tags"} <= names


def test_init_is_idempotent():
    make_lead()
    db.init_db_sync()
    db.init_db_sync()
    assert len(db.list_leads()) == 1  # данные не потерялись


def test_init_creates_missing_parent_directories(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "dir" / "crm.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db_sync()
    assert path.exists()


def test_init_seed_tags():
    db.init_db_sync(seed_tags=["Сайт", "  SEO "])
    tags = db.list_tags()
    assert {t["name"] for t in tags} == {"сайт", "seo"}
    assert all(t["count"] == 0 for t in tags)


def test_connection_pragmas():
    with db._connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


# ---------- normalize_tag ----------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Сайт  ", "сайт"),
        ("SEO   Продвижение", "seo продвижение"),
        ("", ""),
        (None, ""),
        ("а" * 100, "а" * db.MAX_TAG_LEN),
    ],
)
def test_normalize_tag(raw, expected):
    assert db.normalize_tag(raw) == expected


# ---------- create_lead_sync ----------
def test_create_lead_basic_and_defaults():
    lead_id = make_lead()
    lead = db.get_lead(lead_id)

    assert lead["id"] == lead_id
    assert lead["name"] == "Ирина"
    assert lead["contact"] == "@irina_coffee"
    assert lead["request"] == "Сайт для кофейни"
    assert lead["source"] == "manual"
    assert lead["status"] == "new"
    assert lead["tags"] == []
    assert lead["created_at"] and lead["updated_at"]


def test_create_lead_ids_increment():
    assert make_lead() < make_lead()


def test_create_lead_with_tags_normalized_deduped_and_blank_skipped():
    lead_id = make_lead(tags=["Сайт", "сайт", "SEO", "   "])
    assert db.get_lead(lead_id)["tags"] == ["seo", "сайт"]


def test_create_lead_blank_strings_become_none_and_values_stripped():
    lead_id = make_lead(name="   ", contact="  @ivan_dev  ", request="")
    lead = db.get_lead(lead_id)
    assert lead["name"] is None
    assert lead["contact"] == "@ivan_dev"
    assert lead["request"] is None


def test_create_lead_requires_some_content():
    with pytest.raises(ValueError):
        db.create_lead_sync()
    with pytest.raises(ValueError):
        db.create_lead_sync(name="  ", contact="", request=None, raw_text="  ")


def test_create_lead_only_raw_text_is_allowed():
    lead_id = db.create_lead_sync(raw_text="привет", source="bot")
    lead = db.get_lead(lead_id)
    assert lead["raw_text"] == "привет"
    assert lead["name"] is None


def test_create_lead_invalid_source_and_status():
    with pytest.raises(ValueError):
        make_lead(source="email")
    with pytest.raises(ValueError):
        make_lead(status="archived")


def test_create_lead_saves_telegram_fields():
    lead_id = make_lead(source="bot", telegram_id=123456, telegram_username="irina")
    lead = db.get_lead(lead_id)
    assert lead["telegram_id"] == 123456
    assert lead["telegram_username"] == "irina"
    assert lead["source"] == "bot"


def test_create_lead_blank_telegram_username_becomes_none():
    lead_id = make_lead(telegram_username="  ")
    assert db.get_lead(lead_id)["telegram_username"] is None


def test_create_lead_is_atomic_when_tags_fail(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("tags failed")

    monkeypatch.setattr(db, "_add_tags", boom)
    with pytest.raises(RuntimeError):
        make_lead(tags=["сайт"])
    assert db.list_leads() == []  # лид не должен остаться без тегов "наполовину"


# ---------- ограничения на уровне БД ----------
def test_db_check_constraint_rejects_bad_source():
    with pytest.raises(sqlite3.IntegrityError):
        with db._connect() as conn:
            conn.execute("INSERT INTO leads (name, source) VALUES ('x', 'carrier_pigeon')")


def test_db_foreign_key_enforced():
    with pytest.raises(sqlite3.IntegrityError):
        with db._connect() as conn:
            conn.execute("INSERT INTO lead_tags (lead_id, tag_id) VALUES (999, 999)")


# ---------- get_lead ----------
def test_get_lead_missing_returns_none():
    assert db.get_lead(12345) is None


# ---------- list_leads ----------
def test_list_leads_empty():
    assert db.list_leads() == []


def test_list_leads_newest_first():
    ids = [make_lead(name=f"Клиент {i}") for i in range(3)]
    assert [lead["id"] for lead in db.list_leads()] == ids[::-1]


def test_list_leads_filter_by_tag_keeps_all_tags_of_found_leads():
    a = make_lead(name="А", tags=["сайт", "дизайн"])
    make_lead(name="Б", tags=["seo"])
    make_lead(name="В")

    found = db.list_leads(tag="сайт")

    assert [lead["id"] for lead in found] == [a]
    assert found[0]["tags"] == ["дизайн", "сайт"]  # не только искомый тег


def test_list_leads_tag_filter_is_normalized():
    a = make_lead(tags=["сайт"])
    assert [lead["id"] for lead in db.list_leads(tag="  САЙТ ")] == [a]


def test_list_leads_unknown_tag_returns_empty():
    make_lead(tags=["сайт"])
    assert db.list_leads(tag="несуществующий") == []


def test_list_leads_only_untagged():
    make_lead(name="С тегом", tags=["сайт"])
    c = make_lead(name="Без тега")
    found = db.list_leads(only_untagged=True)
    assert [lead["id"] for lead in found] == [c]
    assert found[0]["tags"] == []


def test_list_leads_filter_by_source_and_status():
    bot_id = make_lead(source="bot")
    manual_id = make_lead(source="manual")
    db.update_lead(manual_id, status="done")

    assert [l["id"] for l in db.list_leads(source="bot")] == [bot_id]
    assert [l["id"] for l in db.list_leads(status="done")] == [manual_id]
    assert db.list_leads(source="bot", status="done") == []


def test_list_leads_combined_tag_and_source_filters():
    a = make_lead(source="bot", tags=["сайт"])
    make_lead(source="manual", tags=["сайт"])
    make_lead(source="bot", tags=["seo"])
    assert [l["id"] for l in db.list_leads(tag="сайт", source="bot")] == [a]


@pytest.mark.parametrize("query", ["ирина", "ИРИНА", "ирин"])
def test_search_is_case_insensitive_for_cyrillic(query):
    a = make_lead(name="Ирина")
    make_lead(name="Олег", contact="@oleg_dev", request="Логотип")
    assert [l["id"] for l in db.list_leads(search=query)] == [a]


def test_search_in_contact_request_and_raw_text():
    by_contact = make_lead(name="А", contact="@special_user", request="x")
    by_request = make_lead(name="Б", contact="@b_user1", request="Нужен ЛОГОТИП")
    by_raw = db.create_lead_sync(name="В", raw_text="Хочу таргетированную рекламу")

    assert [l["id"] for l in db.list_leads(search="special")] == [by_contact]
    assert [l["id"] for l in db.list_leads(search="логотип")] == [by_request]
    assert [l["id"] for l in db.list_leads(search="таргетированную")] == [by_raw]


def test_search_no_match_and_blank_search():
    make_lead()
    assert db.list_leads(search="zzz-нет-такого") == []
    assert len(db.list_leads(search="   ")) == 1  # пустой поиск не фильтрует


def test_search_escapes_like_wildcards():
    percent = make_lead(name="А", request="скидка 50%")
    make_lead(name="Б", request="скидка 500")
    underscore = make_lead(name="В", contact="@ivan_dev")
    make_lead(name="Г", contact="@ivanXdev")

    assert [l["id"] for l in db.list_leads(search="50%")] == [percent]
    assert [l["id"] for l in db.list_leads(search="ivan_dev")] == [underscore]


def test_list_leads_limit():
    for i in range(5):
        make_lead(name=f"Клиент {i}")
    assert len(db.list_leads(limit=2)) == 2
    assert len(db.list_leads(limit=0)) == 1  # минимум одна строка
    assert len(db.list_leads(limit=100)) == 5


# ---------- update_lead ----------
def test_update_lead_changes_fields_and_cleans_values():
    lead_id = make_lead()
    assert db.update_lead(lead_id, name="  Анна  ", request="", status="in_progress") is True

    lead = db.get_lead(lead_id)
    assert lead["name"] == "Анна"
    assert lead["request"] is None
    assert lead["status"] == "in_progress"
    assert lead["contact"] == "@irina_coffee"  # остальное не тронуто


def test_update_lead_missing_returns_false():
    assert db.update_lead(999, name="Никто") is False


def test_update_lead_no_fields_returns_false():
    lead_id = make_lead()
    assert db.update_lead(lead_id) is False


def test_update_lead_rejects_unknown_field_and_bad_status():
    lead_id = make_lead()
    with pytest.raises(ValueError):
        db.update_lead(lead_id, source="bot")  # источник менять нельзя
    with pytest.raises(ValueError):
        db.update_lead(lead_id, id=5)
    with pytest.raises(ValueError):
        db.update_lead(lead_id, status="archived")


def test_update_lead_is_safe_against_sql_in_values():
    lead_id = make_lead()
    evil = "x'; DROP TABLE leads; --"
    db.update_lead(lead_id, name=evil)
    assert db.get_lead(lead_id)["name"] == evil
    assert len(db.list_leads()) == 1


# ---------- теги лида ----------
def test_set_lead_tags_replaces_existing():
    lead_id = make_lead(tags=["сайт", "дизайн"])
    assert db.set_lead_tags(lead_id, ["SEO"]) is True
    assert db.get_lead(lead_id)["tags"] == ["seo"]


def test_set_lead_tags_empty_list_clears_tags():
    lead_id = make_lead(tags=["сайт"])
    db.set_lead_tags(lead_id, [])
    assert db.get_lead(lead_id)["tags"] == []


def test_set_lead_tags_missing_lead_returns_false():
    assert db.set_lead_tags(999, ["сайт"]) is False


def test_add_lead_tags_keeps_existing_and_skips_duplicates():
    lead_id = make_lead(tags=["сайт"])
    assert db.add_lead_tags(lead_id, ["Сайт", "SEO"]) is True
    assert db.get_lead(lead_id)["tags"] == ["seo", "сайт"]


def test_add_lead_tags_missing_lead_returns_false():
    assert db.add_lead_tags(999, ["сайт"]) is False


# ---------- delete_lead ----------
def test_delete_lead_removes_lead_and_links_but_keeps_tag():
    lead_id = make_lead(tags=["сайт"])
    assert db.delete_lead(lead_id) is True
    assert db.get_lead(lead_id) is None

    with db._connect() as conn:
        links = conn.execute("SELECT COUNT(*) FROM lead_tags WHERE lead_id = ?", (lead_id,)).fetchone()[0]
    assert links == 0
    assert db.list_tags() == [{"name": "сайт", "count": 0}]


def test_delete_lead_missing_returns_false():
    assert db.delete_lead(999) is False


# ---------- list_tags ----------
def test_list_tags_counts_and_order():
    make_lead(tags=["сайт", "seo"])
    make_lead(tags=["сайт"])
    make_lead(tags=["дизайн"])

    assert db.list_tags() == [
        {"name": "сайт", "count": 2},
        {"name": "seo", "count": 1},
        {"name": "дизайн", "count": 1},
    ]


def test_list_tags_empty():
    assert db.list_tags() == []


# ---------- асинхронные обёртки (для бота) ----------
async def test_async_init_db():
    await db.init_db()
    await db.init_db(seed_tags=["сайт"])
    assert [t["name"] for t in db.list_tags()] == ["сайт"]


async def test_async_create_lead_from_draft():
    d = draft(
        name="Ирина",
        contact="@irina_coffee",
        request="Сайт для кофейни",
        tags=["сайт"],
        raw_text="Привет, я Ирина, нужен сайт для кофейни, @irina_coffee",
    )

    lead_id = await db.create_lead(d, source="bot", telegram_id=777, telegram_username="irina")

    lead = db.get_lead(lead_id)
    assert lead["source"] == "bot"
    assert lead["name"] == "Ирина"
    assert lead["tags"] == ["сайт"]
    assert lead["telegram_id"] == 777
    assert lead["telegram_username"] == "irina"
    assert lead["raw_text"].startswith("Привет")


async def test_async_create_lead_incomplete_draft_is_saved():
    """Принудительное сохранение из бота: полей нет, но переписка есть."""
    lead_id = await db.create_lead(draft(raw_text="Привет"), source="bot", telegram_id=1)
    lead = db.get_lead(lead_id)
    assert lead["name"] is None
    assert lead["raw_text"] == "Привет"


async def test_async_create_lead_empty_draft_raises():
    with pytest.raises(ValueError):
        await db.create_lead(draft(), source="bot")


async def test_async_parallel_creates_do_not_conflict():
    drafts = [draft(name=f"Клиент {i}", raw_text=f"сообщение {i}") for i in range(10)]
    ids = await asyncio.gather(*(db.create_lead(d, source="bot") for d in drafts))

    assert len(set(ids)) == 10
    assert len(db.list_leads()) == 10
