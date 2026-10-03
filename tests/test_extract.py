"""Тесты extract2.py. Сеть не нужна: LLM подменена заглушкой FakeLLM."""
import pytest
from pydantic import ValidationError

from extract2 import (
    ALLOWED_TAGS,
    DEFAULT_TAG,
    MAX_REQUEST_LEN,
    QUESTIONS,
    LeadDraft,
    LLMLead,
    _clean_name,
    _clean_str,
    _contact_in_text,
    _filter_tags,
    _is_greeting,
    _merge,
    _normalize_phone,
    _simple_name,
    _text_without_contacts,
    extract_contacts,
    extract_lead,
    keyword_tags,
    next_missing_field,
    next_question,
)
from openrouter_api import OpenRouterError


# ---------- заглушка LLM ----------
class FakeLLM:
    """Подменяет OpenRouterClient. outcomes: очередь ответов (LLMLead или исключение)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def chat_json(self, system, user, schema, model=None):
        self.calls.append({"system": system, "user": user, "schema": schema})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def llm_lead(name=None, contact=None, request=None, tags=None) -> LLMLead:
    return LLMLead(name=name, contact=contact, request=request, tags=tags or [])


# ---------- extract_contacts ----------
def test_contacts_username():
    assert extract_contacts("Пишите @irina_coffee") == ["@irina_coffee"]


def test_contacts_tme_link_becomes_username():
    assert extract_contacts("Мой тг https://t.me/irina_coffee") == ["@irina_coffee"]


def test_contacts_email_lowercased_and_not_taken_as_username():
    assert extract_contacts("Почта Ivan.Petrov@Example.com") == ["ivan.petrov@example.com"]


@pytest.mark.parametrize(
    "text",
    [
        "Мой номер +7 (916) 123-45-67",
        "Позвоните 8 916 123 45 67",
        "Тел: 89161234567",
        "Позвоните мне 8-916-123-45-67 после 18:00",
    ],
)
def test_contacts_russian_phone_normalized(text):
    assert extract_contacts(text) == ["+79161234567"]


def test_contacts_international_phone():
    assert extract_contacts("Call +44 20 7946 0958") == ["+442079460958"]


def test_contacts_none_found():
    assert extract_contacts("Привет, нужен сайт") == []


def test_contacts_no_false_positive_on_plain_numbers():
    assert extract_contacts("Бюджет 150000 рублей, срок 2 недели") == []
    assert extract_contacts("Номер заказа 1234567890123456") == []


def test_contacts_too_short_username_ignored():
    assert extract_contacts("ник @abc") == []


def test_contacts_deduplicated_case_insensitive():
    assert extract_contacts("@Irina_Coffee и ещё раз https://t.me/irina_coffee") == ["@irina_coffee"]


def test_contacts_multiple_in_expected_order():
    text = "Телефон +7 916 123 45 67, @ivan_dev, mail ivan@mail.ru"
    assert extract_contacts(text) == ["@ivan_dev", "+79161234567", "ivan@mail.ru"]


# ---------- _normalize_phone ----------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("8 (916) 123-45-67", "+79161234567"),
        ("+7 916 123 45 67", "+79161234567"),
        ("+1 (415) 555-2671", "+14155552671"),
        ("12345", None),
        ("+7 999", None),
    ],
)
def test_normalize_phone(raw, expected):
    assert _normalize_phone(raw) == expected


# ---------- _text_without_contacts ----------
def test_text_without_contacts_removes_contact():
    assert _text_without_contacts("Привет, я Ирина. @irina_coffee", ["@irina_coffee"]) == "Привет, я Ирина."


def test_text_without_contacts_only_contact_is_empty():
    assert _text_without_contacts("@irina_coffee", ["@irina_coffee"]) == ""


# ---------- очистка значений от LLM ----------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        (" null ", None),
        ("Не указано", None),
        ("-", None),
        ("«Ирина»", "Ирина"),
        ("  Ирина   Петрова ", "Ирина Петрова"),
    ],
)
def test_clean_str(raw, expected):
    assert _clean_str(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ирина", "Ирина"),
        ("Анна-Мария", "Анна-Мария"),
        ("null", None),
        ("ivan123", None),  # цифры в имени
        ("@ivan_dev", None),  # это контакт, а не имя
        ("А" * 61, None),  # слишком длинное
    ],
)
def test_clean_name(raw, expected):
    assert _clean_name(raw) == expected


# ---------- защита от выдуманных контактов ----------
def test_contact_in_text_username_case_insensitive():
    assert _contact_in_text("@irina", "пишите @IRINA") is True


def test_contact_in_text_phone_by_digits():
    assert _contact_in_text("+79161234567", "звоните +7 916 123-45-67") is True


def test_contact_in_text_fabricated_contact_rejected():
    assert _contact_in_text("+79990001122", "привет, нужен сайт") is False


# ---------- теги ----------
def test_filter_tags_keeps_only_allowed_and_dedupes():
    assert _filter_tags(["Сайт", "SEO", "сайт", "астрология"]) == ["сайт", "seo"]


def test_filter_tags_empty():
    assert _filter_tags([]) == []


def test_keyword_tags_found():
    assert set(keyword_tags("Нужен лендинг и логотип")) == {"сайт", "брендинг"}
    assert set(keyword_tags("Хотим настроить таргет и SEO")) == {"таргет", "seo"}


def test_keyword_tags_none():
    assert keyword_tags("Привет, как дела?") == []


def test_keyword_tags_are_all_allowed():
    assert set(keyword_tags("лендинг логотип таргет seo видео чат-бот")) <= set(ALLOWED_TAGS)


# ---------- LeadDraft / вопросы бота ----------
def test_draft_missing_order_and_completeness():
    d = LeadDraft()
    assert d.missing == ["name", "contact", "request"]
    assert d.is_complete is False

    d = LeadDraft(name="Ирина", contact="@irina_coffee", request="Сайт")
    assert d.missing == []
    assert d.is_complete is True


def test_next_question_follows_missing_fields():
    assert next_question(LeadDraft()) == QUESTIONS["name"]
    assert next_question(LeadDraft(name="Ирина")) == QUESTIONS["contact"]
    assert next_question(LeadDraft(name="Ирина", contact="@irina_coffee")) == QUESTIONS["request"]
    assert next_question(LeadDraft(name="Ирина", contact="@irina_coffee", request="Сайт")) is None


def test_next_missing_field():
    assert next_missing_field(LeadDraft()) == "name"
    assert next_missing_field(LeadDraft(name="Ирина")) == "contact"
    assert next_missing_field(LeadDraft(name="И", contact="@irina_coffee", request="Сайт")) is None


# ---------- схема для LLM ----------
def test_llm_schema_is_strict():
    schema = LLMLead.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "contact", "request", "tags"}


def test_llm_schema_rejects_extra_fields():
    with pytest.raises(ValidationError):
        LLMLead(name=None, contact=None, request=None, tags=[], foo=1)


# ---------- _merge ----------
def test_merge_combines_fields_and_tags():
    old = LeadDraft(name="Ирина", request="Сайт", tags=["сайт"], raw_text="Нужен сайт")
    new = LeadDraft(contact="@irina_coffee", tags=["сайт", "дизайн"], raw_text="Вот контакт")
    merged = _merge(old, new)

    assert merged.name == "Ирина"
    assert merged.contact == "@irina_coffee"
    assert merged.request == "Сайт"
    assert merged.tags == ["сайт", "дизайн"]
    assert merged.raw_text == "Нужен сайт\nВот контакт"


def test_merge_new_value_overrides_old_but_empty_does_not():
    old = LeadDraft(name="Ирина", request="Сайт")
    merged = _merge(old, LeadDraft(request="Сайт и логотип"))
    assert merged.request == "Сайт и логотип"
    assert merged.name == "Ирина"


def test_merge_raw_text_when_old_empty():
    assert _merge(LeadDraft(), LeadDraft(raw_text="Привет")).raw_text == "Привет"


def test_merge_llm_failed_is_sticky():
    merged = _merge(LeadDraft(llm_failed=True), LeadDraft())
    assert merged.llm_failed is True


# ---------- extract_lead: основной сценарий ----------
async def test_extract_lead_full_message():
    text = "Привет, я Ирина, нужен сайт для кофейни, пишите на @irina_coffee"
    llm = FakeLLM(llm_lead(name="Ирина", request="Сайт для кофейни", tags=["сайт"]))

    draft = await extract_lead(llm, text)

    assert draft.name == "Ирина"
    assert draft.contact == "@irina_coffee"
    assert draft.request == "Сайт для кофейни"
    assert draft.tags == ["сайт"]
    assert draft.raw_text == text
    assert draft.llm_failed is False
    assert draft.is_complete is True
    assert len(llm.calls) == 1
    assert llm.calls[0]["schema"] is LLMLead


async def test_extract_lead_prompt_contains_rules_and_delimited_text():
    text = "Меня зовут Олег, нужен дизайн"
    llm = FakeLLM(llm_lead(name="Олег", request="Дизайн", tags=["дизайн"]))

    await extract_lead(llm, text)

    call = llm.calls[0]
    assert all(tag in call["system"] for tag in ALLOWED_TAGS)
    assert "<<<" in call["user"] and ">>>" in call["user"]
    assert text in call["user"]


async def test_extract_lead_only_contact_skips_llm():
    llm = FakeLLM()  # любой вызов упадёт на pop из пустого списка
    old = LeadDraft(name="Ирина", request="Сайт")

    draft = await extract_lead(llm, "@irina_coffee", old, asking="contact")

    assert llm.calls == []
    assert draft.contact == "@irina_coffee"
    assert draft.name == "Ирина"
    assert draft.is_complete is True


# ---------- extract_lead: валидация ответа LLM ----------
async def test_extract_lead_regex_contact_beats_llm_contact():
    llm = FakeLLM(llm_lead(name="Ирина", contact="@other_user", request="Сайт"))
    draft = await extract_lead(llm, "Я Ирина, нужен сайт, мой тг @real_user")
    assert draft.contact == "@real_user"


async def test_extract_lead_hallucinated_contact_dropped():
    llm = FakeLLM(llm_lead(name="Олег", contact="+79990001122", request="Дизайн"))
    draft = await extract_lead(llm, "Меня зовут Олег, нужен дизайн")
    assert draft.contact is None
    assert "contact" in draft.missing


async def test_extract_lead_llm_contact_accepted_if_present_in_text():
    llm = FakeLLM(llm_lead(name="Ирина", contact="irina_coffee", request="Сайт"))
    draft = await extract_lead(llm, "Я Ирина, нужен сайт, мой тг: irina_coffee")
    assert draft.contact == "irina_coffee"


async def test_extract_lead_tags_filtered_to_allowed():
    llm = FakeLLM(llm_lead(request="Сайт", tags=["сайт", "астрология"]))
    draft = await extract_lead(llm, "Нужен сайт для магазина")
    assert draft.tags == ["сайт"]


async def test_extract_lead_garbage_values_cleaned():
    llm = FakeLLM(llm_lead(name="null", request="не указано"))
    draft = await extract_lead(llm, "Здравствуйте, есть вопрос")
    assert draft.name is None
    assert draft.request is None


async def test_extract_lead_name_with_digits_rejected():
    llm = FakeLLM(llm_lead(name="user12345", request="Сайт"))
    draft = await extract_lead(llm, "Нужен сайт, я user12345")
    assert draft.name is None


async def test_extract_lead_request_truncated():
    llm = FakeLLM(llm_lead(request="а" * (MAX_REQUEST_LEN + 500)))
    draft = await extract_lead(llm, "Очень длинное описание задачи")
    assert len(draft.request) == MAX_REQUEST_LEN


# ---------- extract_lead: сбои LLM ----------
async def test_extract_lead_llm_error_falls_back_to_regex_and_keywords():
    text = "Нужен чат-бот для магазина, звоните +7 916 123 45 67"
    llm = FakeLLM(OpenRouterError("boom"))

    draft = await extract_lead(llm, text)

    assert draft.llm_failed is True
    assert draft.contact == "+79161234567"
    assert "чат-бот" in draft.tags
    assert draft.name is None
    assert draft.request is None
    assert draft.missing == ["name", "request"]
    assert draft.raw_text == text  # заявку не теряем


async def test_extract_lead_value_error_also_handled():
    llm = FakeLLM(ValueError("bad json"))
    draft = await extract_lead(llm, "Нужен лендинг, пишите @ivan_dev")
    assert draft.llm_failed is True
    assert draft.contact == "@ivan_dev"
    assert "сайт" in draft.tags


# ---------- extract_lead: слот-филлинг ----------
async def test_extract_lead_slot_filling_llm_only_for_first_message():
    llm = FakeLLM(llm_lead(request="Сайт", tags=["сайт"]))  # LLM нужна только для первого сообщения

    draft = await extract_lead(llm, "Нужен сайт")
    assert draft.missing == ["name", "contact"]
    assert next_missing_field(draft) == "name"

    # короткий ответ на вопрос об имени разбирается кодом
    draft = await extract_lead(llm, "Ирина, @irina_coffee", draft, asking="name")

    assert draft.name == "Ирина"
    assert draft.contact == "@irina_coffee"
    assert draft.request == "Сайт"  # из первого сообщения не потерялось
    assert draft.tags == ["сайт"]
    assert draft.raw_text == "Нужен сайт\nИрина, @irina_coffee"
    assert draft.is_complete is True
    assert len(llm.calls) == 1  # на второе сообщение LLM не вызывалась


async def test_extract_lead_slot_filling_llm_gets_known_data_when_answer_is_not_short():
    llm = FakeLLM(
        llm_lead(request="Сайт", tags=["сайт"]),
        llm_lead(name="Ирина Петровна"),
    )
    draft = await extract_lead(llm, "Нужен сайт")

    # слишком длинное для «имени» (4 слова): решает LLM, и ей передаётся контекст
    draft = await extract_lead(
        llm, "Меня зовут Ирина Петровна Сидорова Младшая, @irina_coffee", draft, asking="name"
    )

    assert draft.name == "Ирина Петровна"
    assert draft.contact == "@irina_coffee"
    assert draft.request == "Сайт"
    assert len(llm.calls) == 2
    second_prompt = llm.calls[1]["user"]
    assert "Сайт" in second_prompt
    assert "name" in second_prompt


# ---------- быстрые ответы без LLM: имя ----------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ирина", "Ирина"),
        ("ирина", "Ирина"),
        ("Меня зовут Олег", "Олег"),
        ("я Олег", "Олег"),
        ("Это Максим!", "Максим"),
        ("Яна", "Яна"),  # «я» внутри слова не принимается за префикс
        ("Анна-Мария", "Анна-Мария"),
        ("Иван Петров", "Иван Петров"),
        ("Олег.", "Олег"),
        ("Ирина,", "Ирина"),
    ],
)
def test_simple_name_accepts(raw, expected):
    assert _simple_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Привет",
        "Добрый день",
        "да",
        "нет",
        "user123",  # цифры
        "@ivan",
        "Как вас зовут?",
        "Хочу заказать сайт",
        "Ирина Петровна Сидорова Младшая",  # больше трёх слов
    ],
)
def test_simple_name_rejects(raw):
    assert _simple_name(raw) is None


async def test_shortcut_name_answer_skips_llm():
    llm = FakeLLM()
    draft = await extract_lead(llm, "Ирина", LeadDraft(), asking="name")

    assert llm.calls == []
    assert draft.name == "Ирина"
    assert draft.missing == ["contact", "request"]


async def test_shortcut_name_with_prefix_skips_llm():
    llm = FakeLLM()
    draft = await extract_lead(llm, "меня зовут ирина", LeadDraft(), asking="name")
    assert llm.calls == []
    assert draft.name == "Ирина"


async def test_shortcut_name_and_contact_in_one_short_answer():
    llm = FakeLLM()
    draft = await extract_lead(llm, "Ирина, @irina_coffee", LeadDraft(), asking="name")
    assert llm.calls == []
    assert draft.name == "Ирина"
    assert draft.contact == "@irina_coffee"


async def test_name_question_answered_with_a_sentence_goes_to_llm():
    llm = FakeLLM(llm_lead(request="Сайт для кофейни", tags=["сайт"]))
    draft = await extract_lead(llm, "Хочу заказать сайт для кофейни", LeadDraft(), asking="name")
    assert len(llm.calls) == 1
    assert draft.name is None
    assert draft.request == "Сайт для кофейни"


async def test_name_question_answered_with_greeting_goes_to_llm():
    llm = FakeLLM(llm_lead())
    draft = await extract_lead(llm, "Привет", LeadDraft(), asking="name")
    assert len(llm.calls) == 1
    assert draft.name is None


async def test_name_question_answered_with_only_contact_skips_llm():
    llm = FakeLLM()
    draft = await extract_lead(llm, "@irina_coffee", LeadDraft(), asking="name")
    assert llm.calls == []
    assert draft.contact == "@irina_coffee"
    assert draft.name is None


# ---------- быстрые ответы без LLM: запрос ----------
async def test_shortcut_request_answer_uses_text_and_keyword_tags():
    llm = FakeLLM()
    old = LeadDraft(name="Ирина", contact="@irina_coffee")

    draft = await extract_lead(llm, "Нужен чат-бот для магазина", old, asking="request")

    assert llm.calls == []
    assert draft.request == "Нужен чат-бот для магазина"
    assert draft.tags == ["чат-бот"]
    assert draft.is_complete is True


async def test_shortcut_request_without_keywords_gets_default_tag():
    llm = FakeLLM()
    old = LeadDraft(name="Ирина", contact="@irina_coffee")

    draft = await extract_lead(llm, "Хочу праздничную акцию", old, asking="request")

    assert llm.calls == []
    assert draft.tags == [DEFAULT_TAG]


async def test_shortcut_request_without_keywords_keeps_existing_tags_instead_of_default():
    llm = FakeLLM()
    old = LeadDraft(name="Ирина", contact="@irina_coffee", tags=["сайт"])

    draft = await extract_lead(llm, "Хочу праздничную акцию", old, asking="request")

    assert draft.tags == ["сайт"]


async def test_shortcut_request_also_picks_up_contact_from_the_answer():
    llm = FakeLLM()
    old = LeadDraft(name="Ирина")

    draft = await extract_lead(llm, "Нужен сайт, пишите +7 916 123 45 67", old, asking="request")

    assert llm.calls == []
    assert draft.contact == "+79161234567"
    assert draft.request.startswith("Нужен сайт")
    assert "сайт" in draft.tags


async def test_request_answer_too_short_goes_to_llm():
    llm = FakeLLM(llm_lead())
    old = LeadDraft(name="Ирина", contact="@irina_coffee")
    await extract_lead(llm, "ок", old, asking="request")
    assert len(llm.calls) == 1


# ---------- быстрые ответы без LLM: контакт ----------
async def test_shortcut_contact_tg_hint_without_at():
    llm = FakeLLM()
    old = LeadDraft(name="Ирина", request="Сайт")

    draft = await extract_lead(llm, "tg: irina_coffee", old, asking="contact")

    assert llm.calls == []
    assert draft.contact == "@irina_coffee"
    assert draft.is_complete is True


async def test_shortcut_contact_short_answer_without_contact_skips_llm():
    llm = FakeLLM()
    old = LeadDraft(name="Ирина", request="Сайт")

    draft = await extract_lead(llm, "не хочу давать", old, asking="contact")

    assert llm.calls == []
    assert draft.contact is None
    assert draft.missing == ["contact"]
    assert draft.raw_text.endswith("не хочу давать")


async def test_long_answer_to_contact_question_goes_to_llm():
    llm = FakeLLM(llm_lead())
    old = LeadDraft(name="Ирина", request="Сайт")
    await extract_lead(llm, "Мне удобнее, чтобы вы позвонили мне вечером, после шести часов", old, asking="contact")
    assert len(llm.calls) == 1


# ---------- быстрые ответы без LLM: приветствие ----------
@pytest.mark.parametrize("text", ["Привет", "Привет!", "Добрый день", "здравствуйте"])
def test_is_greeting_true(text):
    assert _is_greeting(text) is True


@pytest.mark.parametrize("text", ["", "Привет, нужен сайт", "Привет 123", "Нужен сайт"])
def test_is_greeting_false(text):
    assert _is_greeting(text) is False


async def test_greeting_as_first_message_skips_llm():
    llm = FakeLLM()
    draft = await extract_lead(llm, "Привет!")
    assert llm.calls == []
    assert draft.missing == ["name", "contact", "request"]
    assert draft.raw_text == "Привет!"


async def test_greeting_with_task_goes_to_llm():
    llm = FakeLLM(llm_lead(request="Сайт", tags=["сайт"]))
    await extract_lead(llm, "Привет, нужен сайт")
    assert len(llm.calls) == 1


async def test_first_long_message_always_goes_to_llm():
    llm = FakeLLM(llm_lead(name="Ирина", request="Сайт для кофейни", tags=["сайт"]))
    draft = await extract_lead(llm, "Привет, я Ирина, нужен сайт для кофейни, @irina_coffee")
    assert len(llm.calls) == 1
    assert draft.is_complete is True


# ---------- extract_lead: пустой ввод ----------
@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
async def test_extract_lead_empty_text_returns_draft_unchanged(text):
    llm = FakeLLM()
    old = LeadDraft(name="Ирина")

    result = await extract_lead(llm, text, old)

    assert result is old
    assert llm.calls == []
