FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    DB_PATH=/data/crm.db

WORKDIR /app

# tzdata нужен для часовых поясов (APP_TZ), poetry для установки зависимостей из pyproject
RUN pip install poetry tzdata

# зависимости ставим отдельным слоем: он кэшируется, пока не меняются pyproject.toml и poetry.lock
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

COPY . .

# База лежит в /data. К этому каталогу на хостинге ОБЯЗАТЕЛЬНО подключить постоянный том,
# иначе crm.db будет потеряна при каждом перезапуске контейнера
RUN mkdir -p /data

EXPOSE 8501

# start.py запускает и бота, и Streamlit; порт берётся из переменной PORT (по умолчанию 8501)
CMD ["python", "start.py"]