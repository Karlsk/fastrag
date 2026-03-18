from __future__ import annotations

import re

from fastrag.cleaner.base import BaseCleaner


class RegexCleaner(BaseCleaner):
    """Apply user-defined regex substitutions."""

    def __init__(self, patterns: list[tuple[str, str]] | None = None):
        """
        Args:
            patterns: List of (pattern, replacement) tuples.
        """
        self._patterns = [(re.compile(p), r) for p, r in (patterns or [])]

    def clean(self, text: str) -> str:
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return text
