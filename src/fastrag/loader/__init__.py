from __future__ import annotations

from pathlib import Path

from fastrag.loader.base import BaseLoader
from fastrag.loader.csv_excel import CSVLoader, ExcelLoader
from fastrag.loader.docx import DocxLoader
from fastrag.loader.html import HTMLLoader
from fastrag.loader.markdown import MarkdownLoader
from fastrag.loader.pdf import PDFLoader
from fastrag.loader.txt import TxtLoader

__all__ = [
    "BaseLoader",
    "TxtLoader",
    "PDFLoader",
    "DocxLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "CSVLoader",
    "ExcelLoader",
    "get_loader",
]

_LOADER_MAP: dict[str, type[BaseLoader]] = {}


def _register_loaders() -> None:
    for cls in (TxtLoader, PDFLoader, DocxLoader, MarkdownLoader, HTMLLoader, CSVLoader, ExcelLoader):
        for ext in cls.supported_extensions:
            _LOADER_MAP[ext.lower()] = cls


_register_loaders()


def get_loader(filename: str) -> BaseLoader:
    """Return an appropriate loader instance based on file extension."""
    ext = Path(filename).suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: {sorted(_LOADER_MAP.keys())}")
    return loader_cls()
