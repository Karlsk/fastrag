from __future__ import annotations

import re

from fastrag.cleaner.base import BaseCleaner


class WhitespaceCleaner(BaseCleaner):
    """Normalize whitespace: collapse multiple spaces/newlines, strip leading/trailing."""

    def __init__(self, preserve_newlines: bool = False):
        self._preserve_newlines = preserve_newlines

    def clean(self, text: str) -> str:
        if self._preserve_newlines:
            lines = text.splitlines()
            lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in lines]
            text = "\n".join(lines)
            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            text = re.sub(r"\s+", " ", text)
        return text.strip()
