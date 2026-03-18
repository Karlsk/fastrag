from __future__ import annotations

import unicodedata

from fastrag.cleaner.base import BaseCleaner


class ControlCharRemover(BaseCleaner):
    """Remove control characters (except newline and tab)."""

    def clean(self, text: str) -> str:
        return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))
