import logging  # Стандартная библиотека Python для логирования
import os  # Библиотека для работы с операционной системой и файлами
import re  # Проверка имени файла лога
import sys  # Имя запущенной программы (для имени файла лога)
from logging.handlers import RotatingFileHandler  # Файл лога с ротацией по размеру
from pathlib import Path

import colorlog  # Для цвета логов
from dotenv import load_dotenv  # Чтобы настройки логов можно было задать в .env

# Уровень логирования приложения.
# Варианты: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOG_LEVEL = logging.INFO

# Настройки файлов логов (читаются из переменных окружения при создании AppLogger):
#   LOG_DIR           - каталог логов (по умолчанию logs; в контейнере /data/logs на постоянном томе)
#   LOG_MAX_MB        - максимальный размер одного файла в МБ (по умолчанию 5)
#   LOG_BACKUP_COUNT  - сколько старых файлов хранить (по умолчанию 5)
#   LOG_TO_FILE       - 0/false отключает запись в файл (останется только консоль)
#   LOG_NAME          - имя файла лога без расширения (по умолчанию по имени программы: bot, app...)
# Предел занимаемого места: LOG_MAX_MB * (LOG_BACKUP_COUNT + 1) на каждый процесс.
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_MAX_MB = 5.0
DEFAULT_LOG_BACKUP_COUNT = 5


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value >= 0 else default
    except (TypeError, ValueError):
        return default


def _process_log_name() -> str:
    """Имя файла лога. У каждого процесса оно своё: ротацию одного файла двумя процессами
    одновременно выполнять небезопасно (один переименует файл, пока другой в него пишет)."""
    explicit = (os.getenv("LOG_NAME") or "").strip()
    candidate = explicit
    if not candidate and sys.argv and sys.argv[0]:
        path = Path(sys.argv[0])
        candidate = path.parent.name if path.stem == "__main__" else path.stem
    return candidate if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", candidate or "") else "chat_app"


class AppLogger:
    """
    Класс для логирования работы приложения.

    Обеспечивает:
    - Сохранение логов в файлы с ротацией по размеру (диск не заполняется)
    - Вывод логов в консоль
    - Различные уровни логирования (debug, info, warning, error)
    - Форматирование сообщений с временными метками
    """

    def __init__(self):
        """
        Инициализация системы логирования.

        Настраивает:
        - Директорию для хранения логов
        - Форматирование сообщений
        - Обработчики для файла (с ротацией) и консоли
        - Уровни логирования
        """
        load_dotenv()  # настройки логов могут лежать в .env (уже заданные переменные не перезаписываются)

        # Настройка основного логгера приложения
        self.logger = logging.getLogger('ChatApp')  # Создание логгера с именем
        self.logger.setLevel(LOG_LEVEL)  # Установка уровня логирования

        # Обработчики добавляются один раз на процесс (повторные AppLogger() используют те же)
        if not self.logger.handlers:
            # Цветной формат — только для консоли
            console_formatter = colorlog.ColoredFormatter(
                "%(yellow)s%(asctime)s%(reset)s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler = colorlog.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            file_handler = self._build_file_handler()
            if file_handler is not None:
                self.logger.addHandler(file_handler)

        self.logs_dir = os.getenv("LOG_DIR", DEFAULT_LOG_DIR)

    @staticmethod
    def _build_file_handler():
        """Файловый обработчик с ротацией. Если писать в файл нельзя, логи идут только в консоль."""
        if os.getenv("LOG_TO_FILE", "1").strip().lower() in ("0", "false", "no", "off"):
            return None

        logs_dir = Path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            max_bytes = int(_env_float("LOG_MAX_MB", DEFAULT_LOG_MAX_MB) * 1024 * 1024)
            handler = RotatingFileHandler(
                logs_dir / f"{_process_log_name()}.log",
                maxBytes=max_bytes,  # при превышении файл переименовывается в .log.1, .log.2 ...
                backupCount=_env_int("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT),  # старше удаляются
                encoding='utf-8',
            )
        except OSError as e:
            print(f"Не удалось открыть файл лога в '{logs_dir}': {e}. Логи пишутся только в консоль.", file=sys.stderr)
            return None

        # Обычный (не цветной) формат — для файла
        # Формат: YYYY-MM-DD HH:MM:SS - LEVEL - Message
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        )
        return handler

    def info(self, message: str):
        """
        Логирование информационного сообщения.

        Используется для записи важной информации о работе приложения:
        - Успешные операции
        - Статус выполнения
        - Информация о состоянии

        Args:
            message (str): Текст информационного сообщения
        """
        self.logger.info(message)

    def error(self, message: str, exc_info=None):
        """
        Логирование ошибки.

        Используется для записи информации об ошибках:
        - Исключения
        - Сбои в работе
        - Критические ошибки

        Args:
            message (str): Текст сообщения об ошибке
            exc_info: Информация об исключении (по умолчанию None)
                     Если передано True, автоматически добавляет стек вызовов
        """
        self.logger.error(message, exc_info=exc_info)

    def debug(self, message: str):
        """
        Логирование отладочной информации.

        Используется для записи подробной информации для отладки:
        - Значения переменных
        - Промежуточные результаты
        - Детали выполнения

        Args:
            message (str): Текст отладочного сообщения
        """
        self.logger.debug(message)

    def warning(self, message: str):
        """
        Логирование предупреждения.

        Используется для записи предупреждений:
        - Потенциальные проблемы
        - Нежелательные ситуации
        - Предупреждения о состоянии

        Args:
            message (str): Текст предупреждения
        """
        self.logger.warning(message)
