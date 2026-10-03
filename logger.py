import logging  # Стандартная библиотека Python для логирования
import colorlog # Для цвета логов
import os  # Библиотека для работы с операционной системой и файлами
from datetime import datetime  # Библиотека для работы с датой и временем

# Уровень логирования приложения.
# Варианты: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
LOG_LEVEL = logging.INFO


class AppLogger:
    """
    Класс для логирования работы приложения.

    Обеспечивает:
    - Сохранение логов в файлы с датой в имени
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
        - Обработчики для файла и консоли
        - Уровни логирования
        """
        # Создание директории для хранения файлов логов
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        # Формирование имени файла лога с текущей датой
        # Формат: chat_app_YYYY-MM-DD.log
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.logs_dir, f"chat_app_{current_date}.log")

        # Настройка основного логгера приложения
        self.logger = logging.getLogger('ChatApp')  # Создание логгера с именем
        self.logger.setLevel(LOG_LEVEL)  # Установка уровня логирования

        # Настройка формата сообщений лога
        # Формат: YYYY-MM-DD HH:MM:SS - LEVEL - Message
        if not self.logger.handlers:
            # Обычный (не цветной) формат — для файла
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(file_formatter)

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

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

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