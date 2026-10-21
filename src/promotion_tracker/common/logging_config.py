"""Configuração compartilhada dos logs da aplicação."""

import logging
import sys
from datetime import date, datetime
from pathlib import Path

from .paths import PROJECT_PATHS


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_MARKER = "_promotion_tracker_handler"


class DailyFileHandler(logging.Handler):
    """Grava logs em um arquivo por dia, organizado em diretórios anuais/mensais."""

    def __init__(self, logs_dir: Path) -> None:
        super().__init__()
        self.logs_dir = logs_dir
        self._current_date: date | None = None
        self._file_handler: logging.FileHandler | None = None

    def emit(self, record: logging.LogRecord) -> None:
        record_date = datetime.fromtimestamp(record.created).date()
        if record_date != self._current_date:
            self._open_file(record_date)
        if self._file_handler is not None:
            self._file_handler.emit(record)

    def _open_file(self, log_date: date) -> None:
        if self._file_handler is not None:
            self._file_handler.close()

        log_dir = self.logs_dir / f"{log_date:%Y}" / f"{log_date:%m}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"promotion_tracker_{log_date:%Y-%m-%d}.log"
        self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._file_handler.setFormatter(self.formatter)
        self._current_date = log_date

    def close(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None
        super().close()


def configure_logging(debug: bool = False) -> None:
    """Exibe o progresso no terminal e grava logs diários por ano e mês."""
    logs_dir = PROJECT_PATHS.root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if getattr(handler, _HANDLER_MARKER, None) == "console"
        ),
        None,
    )
    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        setattr(console_handler, _HANDLER_MARKER, "console")
        root_logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    )

    file_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if getattr(handler, _HANDLER_MARKER, None) == "file"
        ),
        None,
    )
    if file_handler is None:
        file_handler = DailyFileHandler(logs_dir)
        setattr(file_handler, _HANDLER_MARKER, "file")
        root_logger.addHandler(file_handler)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    )

    for name in ("httpx", "httpcore", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
