# ---------- этап 1: установка зависимостей (Poetry и его кэш остаются здесь и в итоговый образ не попадают) ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

RUN pip install poetry

# зависимости ставятся отдельным слоем: он кэшируется, пока не меняются pyproject.toml и poetry.lock
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

# tzdata нужен для часовых поясов (APP_TZ); ставим прямо в окружение проекта
RUN pip install --target /app/.venv/lib/python3.12/site-packages tzdata


# ---------- этап 2: итоговый образ (только Python, готовое окружение и код) ----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_PATH=/data/crm.db \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

# База лежит в /data. К этому каталогу на хостинге ОБЯЗАТЕЛЬНО подключить постоянный том,
# иначе crm.db будет потеряна при каждом перезапуске контейнера
RUN mkdir -p /data

EXPOSE 8501

# start.py запускает и бота, и Streamlit; порт берётся из переменной PORT (по умолчанию 8501)
CMD ["python", "start.py"]