"""FastRAG - A modular RAG engine with extensible document processing, retrieval, and low-code pipeline."""

__version__ = "0.1.0"

# Auto-initialize logging on import
from fastrag.config import get_settings as _get_settings
from fastrag.logging import setup_logging as _setup_logging

_s = _get_settings()
_setup_logging(level=_s.LOG_LEVEL, log_dir=_s.LOG_DIR)
del _s
