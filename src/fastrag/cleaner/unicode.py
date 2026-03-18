from __future__ import annotations

import unicodedata

from fastrag.cleaner.base import BaseCleaner


class UnicodeNormalizer(BaseCleaner):
    """Normalize Unicode text using NFKC form. Converts fullwidth to halfwidth."""

    def __init__(self, form: str = "NFKC"):
        self._form = form

    def clean(self, text: str) -> str:
        return unicodedata.normalize(self._form, text)
