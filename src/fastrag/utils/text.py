from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_control_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t"))


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)
