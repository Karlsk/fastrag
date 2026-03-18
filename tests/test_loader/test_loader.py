from pathlib import Path

from fastrag.loader import TxtLoader, MarkdownLoader, get_loader
from fastrag.loader.base import BaseLoader


class TestTxtLoader:
    def test_load_from_file(self, sample_txt_file):
        loader = TxtLoader()
        docs = loader.load(sample_txt_file)
        assert len(docs) == 1
        assert "FastRAG" in docs[0].content
        assert docs[0].metadata.file_type == "txt"
        assert docs[0].metadata.source == str(sample_txt_file)

    def test_load_from_bytes(self):
        loader = TxtLoader()
        docs = loader.load(b"Hello world")
        assert len(docs) == 1
        assert docs[0].content == "Hello world"
        assert docs[0].metadata.source == "<bytes>"

    def test_supported_extensions(self):
        assert ".txt" in TxtLoader.supported_extensions


class TestMarkdownLoader:
    def test_load_from_file(self, sample_md_file):
        loader = MarkdownLoader()
        docs = loader.load(sample_md_file)
        assert len(docs) == 1
        assert "FastRAG" in docs[0].content
        assert docs[0].metadata.file_type == "markdown"
        headers = docs[0].metadata.extra.get("headers", [])
        assert len(headers) >= 2
        assert headers[0]["text"] == "Features"


class TestGetLoader:
    def test_get_loader_txt(self):
        loader = get_loader("test.txt")
        assert isinstance(loader, TxtLoader)

    def test_get_loader_md(self):
        loader = get_loader("test.md")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_unsupported(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported file extension"):
            get_loader("test.xyz")

    def test_all_loaders_are_base_loader(self):
        for ext in [".txt", ".md", ".pdf", ".html", ".csv", ".xlsx", ".docx"]:
            loader = get_loader(f"test{ext}")
            assert isinstance(loader, BaseLoader)
