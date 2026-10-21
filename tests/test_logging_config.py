import logging
from datetime import datetime, timedelta

from promotion_tracker.common import logging_config
from promotion_tracker.common.paths import ProjectPaths


def test_configure_logging_writes_debug_to_file_and_progress_to_console(
    tmp_path, monkeypatch, capsys
):
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level
    preserved_handlers = []
    for handler in original_handlers:
        if getattr(handler, logging_config._HANDLER_MARKER, None):
            root_logger.removeHandler(handler)
            handler.close()
        else:
            preserved_handlers.append(handler)
    monkeypatch.setattr(
        logging_config,
        "PROJECT_PATHS",
        ProjectPaths(root=tmp_path),
    )

    try:
        logging_config.configure_logging()
        test_logger = logging.getLogger("promotion_tracker.test")
        test_logger.debug("detalhe de diagnóstico")
        test_logger.info("etapa iniciada")

        console_output = capsys.readouterr().out
        today = datetime.now().date()
        log_file = (
            tmp_path
            / "logs"
            / f"{today:%Y}"
            / f"{today:%m}"
            / f"promotion_tracker_{today:%Y-%m-%d}.log"
        )
        file_output = log_file.read_text(encoding="utf-8")

        assert "etapa iniciada" in console_output
        assert "detalhe de diagnóstico" not in console_output
        assert "etapa iniciada" in file_output
        assert "detalhe de diagnóstico" in file_output

        file_handler = next(
            handler
            for handler in root_logger.handlers
            if getattr(handler, logging_config._HANDLER_MARKER, None) == "file"
        )
        next_day = datetime.now() + timedelta(days=1)
        next_day_record = test_logger.makeRecord(
            test_logger.name,
            logging.INFO,
            __file__,
            1,
            "log do dia seguinte",
            (),
            None,
        )
        next_day_record.created = next_day.timestamp()
        file_handler.handle(next_day_record)
        next_day_path = (
            tmp_path
            / "logs"
            / f"{next_day:%Y}"
            / f"{next_day:%m}"
            / f"promotion_tracker_{next_day:%Y-%m-%d}.log"
        )
        assert "log do dia seguinte" in next_day_path.read_text(
            encoding="utf-8"
        )
    finally:
        for handler in root_logger.handlers.copy():
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()
        for handler in preserved_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
