"""Centralized logging for FastRAG.

Configures the ``fastrag`` logger namespace with console and optional
rotating-file handlers.  All sub-modules that use
``logger = logging.getLogger(__name__)`` automatically inherit this
configuration.

Usage::

    # Recommended — per-module logger (already used by most modules)
    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")

    # Alternative — static utility with auto caller detection
    from fastrag.logging import FastRAGLogger
    FastRAGLogger.info("something happened")
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_initialized = False

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class LevelFilter(logging.Filter):
    """Allow only records within *[min_level, max_level]*."""

    def __init__(self, min_level: int, max_level: int) -> None:
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return self.min_level <= record.levelno <= self.max_level


def setup_logging(level: str = "INFO", log_dir: str = "") -> None:
    """Configure the ``fastrag`` logger namespace.

    This function is **idempotent** — calling it more than once is safe but
    only the first invocation takes effect.

    Parameters
    ----------
    level:
        Minimum log level for the console handler.  One of
        ``DEBUG / INFO / WARNING / ERROR / CRITICAL``.
    log_dir:
        Directory for rotating log files.  When empty (the default) only
        the console handler is active.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    root = logging.getLogger("fastrag")
    root.setLevel(logging.DEBUG)  # capture everything; handlers do filtering
    root.propagate = False  # don't leak into host (Streamlit) root logger

    # --- Console handler (stderr — unbuffered, visible in docker logs) ---
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # --- File handlers (optional) ---
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            print(f"[fastrag] WARNING: cannot create log dir '{log_dir}': {exc}", file=sys.stderr)
            return

        file_specs: list[tuple[str, int, int, int, int]] = [
            # (filename, min_level, max_level, maxBytes, backupCount)
            ("debug.log", logging.DEBUG, logging.DEBUG, 10 * 1024 * 1024, 5),
            ("info.log", logging.INFO, logging.INFO, 10 * 1024 * 1024, 5),
            ("warn.log", logging.WARNING, logging.WARNING, 5 * 1024 * 1024, 3),
            ("error.log", logging.ERROR, logging.CRITICAL, 5 * 1024 * 1024, 3),
        ]

        for fname, min_lvl, max_lvl, max_bytes, backup in file_specs:
            handler = RotatingFileHandler(
                os.path.join(log_dir, fname),
                maxBytes=max_bytes,
                backupCount=backup,
                encoding="utf-8",
            )
            handler.setLevel(min_lvl)
            handler.setFormatter(formatter)
            handler.addFilter(LevelFilter(min_lvl, max_lvl))
            root.addHandler(handler)


class FastRAGLogger:
    """Static utility class for convenient logging.

    Automatically detects the caller's module name and routes to the
    corresponding ``fastrag.*`` logger.

    Example::

        FastRAGLogger.info("document loaded", source="test.pdf", pages=10)
    """

    @staticmethod
    def _get_logger() -> logging.Logger:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back.f_back  # type: ignore[union-attr]
            name = caller.f_globals.get("__name__", "fastrag")  # type: ignore[union-attr]
        finally:
            del frame
        return logging.getLogger(name)

    @staticmethod
    def _format_msg(msg: str, kwargs: dict) -> str:
        if kwargs:
            pairs = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{msg} {pairs}"
        return msg

    @staticmethod
    def debug(msg: str, *args: object, **kwargs: object) -> None:
        logger = FastRAGLogger._get_logger()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(FastRAGLogger._format_msg(msg, kwargs), *args)

    @staticmethod
    def info(msg: str, *args: object, **kwargs: object) -> None:
        logger = FastRAGLogger._get_logger()
        if logger.isEnabledFor(logging.INFO):
            logger.info(FastRAGLogger._format_msg(msg, kwargs), *args)

    @staticmethod
    def warning(msg: str, *args: object, **kwargs: object) -> None:
        logger = FastRAGLogger._get_logger()
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(FastRAGLogger._format_msg(msg, kwargs), *args)

    @staticmethod
    def error(msg: str, *args: object, exc_info: Optional[bool] = None, **kwargs: object) -> None:
        logger = FastRAGLogger._get_logger()
        if logger.isEnabledFor(logging.ERROR):
            logger.error(FastRAGLogger._format_msg(msg, kwargs), *args, exc_info=exc_info)

    @staticmethod
    def exception(msg: str, *args: object, **kwargs: object) -> None:
        logger = FastRAGLogger._get_logger()
        if logger.isEnabledFor(logging.ERROR):
            logger.exception(FastRAGLogger._format_msg(msg, kwargs), *args)

    @staticmethod
    def critical(msg: str, *args: object, **kwargs: object) -> None:
        logger = FastRAGLogger._get_logger()
        if logger.isEnabledFor(logging.CRITICAL):
            logger.critical(FastRAGLogger._format_msg(msg, kwargs), *args)
