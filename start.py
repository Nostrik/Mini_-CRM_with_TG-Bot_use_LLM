"""Запускает бота и Streamlit-CRM одной командой (для контейнера и хостинга).

    python start.py

Оба процесса работают параллельно и пишут логи в общий вывод. Если один из них упал,
скрипт останавливает второй и завершается с ошибкой: платформа (Docker restart policy,
Railway, systemd и т.п.) перезапустит контейнер целиком, и оба процесса поднимутся заново.

Переменные окружения:
    PORT  - порт веб-интерфейса (хостинги задают его сами), по умолчанию 8501
    HOST  - адрес, на котором слушает Streamlit, по умолчанию 0.0.0.0
    DB_PATH - путь к SQLite (общий для бота и CRM, на хостинге: файл на постоянном томе)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

PORT = os.getenv("PORT", "8501")
HOST = os.getenv("HOST", "0.0.0.0")
STOP_TIMEOUT_SEC = 10


def log(message: str) -> None:
    print(f"[start] {message}", flush=True)


def build_commands() -> dict[str, list[str]]:
    return {
        "bot": [sys.executable, "bot.py"],
        "crm": [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", PORT,
            "--server.address", HOST,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
    }


def warn_if_db_not_persistent() -> None:
    """В контейнере БД должна лежать на подключённом томе, иначе она пропадёт при перезапуске."""
    db_path = os.getenv("DB_PATH", "crm.db")
    folder = os.path.dirname(os.path.abspath(db_path))
    if os.name == "posix" and folder == "/data" and not os.path.ismount("/data"):
        log("ВНИМАНИЕ: каталог /data не является подключённым томом. "
            "База crm.db будет потеряна при перезапуске контейнера. Подключите том к /data.")


def terminate_all(procs: dict[str, subprocess.Popen]) -> None:
    """Мягко останавливает процессы, а тех, кто не успел за STOP_TIMEOUT_SEC, убивает."""
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + STOP_TIMEOUT_SEC
    for proc in procs.values():
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    warn_if_db_not_persistent()

    stopping = False

    def request_stop(signum, frame) -> None:  # noqa: ARG001
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    procs: dict[str, subprocess.Popen] = {}
    for name, command in build_commands().items():
        # свой файл лога у каждого процесса (bot.log, crm.log): ротация одного файла двумя процессами небезопасна
        procs[name] = subprocess.Popen(command, env={**os.environ, "LOG_NAME": name})
        log(f"запущен процесс '{name}' (pid {procs[name].pid})")
    log(f"CRM доступна на порту {PORT}")

    try:
        while True:
            if stopping:
                log("получен сигнал остановки, завершаю процессы")
                terminate_all(procs)
                return 0

            for name, proc in procs.items():
                code = proc.poll()
                if code is not None:
                    log(f"процесс '{name}' завершился с кодом {code}, останавливаю остальные")
                    terminate_all(procs)
                    return code or 1

            time.sleep(1)
    finally:
        terminate_all(procs)


if __name__ == "__main__":
    sys.exit(main())
