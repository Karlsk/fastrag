from __future__ import annotations

import re

from fastrag.cleaner.base import BaseCleaner


class HTMLCleaner(BaseCleaner):
    """Remove HTML tags from text."""

    _TAG_RE = re.compile(r"<[^>]+>")
    _COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    _ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ", "&quot;": '"', "&#39;": "'"}

    def clean(self, text: str) -> str:
        text = self._COMMENT_RE.sub("", text)
        text = self._TAG_RE.sub("", text)
        for entity, char in self._ENTITY_MAP.items():
            text = text.replace(entity, char)
        return text
