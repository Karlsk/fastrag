from __future__ import annotations

import logging
from abc import ABC, abstractmethod


class BaseCleaner(ABC):
    """Abstract base class for text cleaners.

    Subclasses must implement `clean()`. Cleaners can be chained using the `+` operator.
    """

    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean the input text and return the cleaned result."""
        ...

    def __add__(self, other: BaseCleaner) -> CleanerPipeline:
        if isinstance(other, CleanerPipeline):
            return CleanerPipeline([self, *other.cleaners])
        return CleanerPipeline([self, other])


logger = logging.getLogger(__name__)


class CleanerPipeline(BaseCleaner):
    """Chain multiple cleaners together. Executes them in order."""

    def __init__(self, cleaners: list[BaseCleaner] | None = None):
        self.cleaners: list[BaseCleaner] = cleaners or []

    def clean(self, text: str) -> str:
        orig_len = len(text)
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        logger.info(
            "CleanerPipeline: %d cleaners, %d -> %d chars",
            len(self.cleaners), orig_len, len(text),
        )
        return text

    def __add__(self, other: BaseCleaner) -> CleanerPipeline:
        if isinstance(other, CleanerPipeline):
            return CleanerPipeline([*self.cleaners, *other.cleaners])
        return CleanerPipeline([*self.cleaners, other])

    def __repr__(self) -> str:
        names = [type(c).__name__ for c in self.cleaners]
        return f"CleanerPipeline({' -> '.join(names)})"
