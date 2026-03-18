from fastrag.cleaner import (
    BaseCleaner,
    CleanerPipeline,
    ControlCharRemover,
    HTMLCleaner,
    RegexCleaner,
    UnicodeNormalizer,
    WhitespaceCleaner,
)


class TestHTMLCleaner:
    def test_remove_tags(self):
        cleaner = HTMLCleaner()
        assert cleaner.clean("<p>Hello <b>world</b></p>") == "Hello world"

    def test_remove_comments(self):
        cleaner = HTMLCleaner()
        assert cleaner.clean("Hello <!-- comment --> world") == "Hello  world"

    def test_decode_entities(self):
        cleaner = HTMLCleaner()
        assert cleaner.clean("&amp; &lt; &gt;") == "& < >"


class TestWhitespaceCleaner:
    def test_collapse_spaces(self):
        cleaner = WhitespaceCleaner()
        assert cleaner.clean("hello   world") == "hello world"

    def test_collapse_newlines(self):
        cleaner = WhitespaceCleaner()
        assert cleaner.clean("hello\n\n\n\nworld") == "hello world"

    def test_preserve_newlines(self):
        cleaner = WhitespaceCleaner(preserve_newlines=True)
        result = cleaner.clean("hello\n\n\n\nworld")
        assert result == "hello\n\nworld"


class TestUnicodeNormalizer:
    def test_nfkc(self):
        cleaner = UnicodeNormalizer()
        assert cleaner.clean("\uff21\uff22\uff23") == "ABC"


class TestControlCharRemover:
    def test_remove_control(self):
        cleaner = ControlCharRemover()
        result = cleaner.clean("hello\x00\x01world")
        assert result == "helloworld"

    def test_keep_newline_tab(self):
        cleaner = ControlCharRemover()
        assert cleaner.clean("hello\n\tworld") == "hello\n\tworld"


class TestRegexCleaner:
    def test_custom_pattern(self):
        cleaner = RegexCleaner(patterns=[(r"\d+", "NUM")])
        assert cleaner.clean("There are 42 items") == "There are NUM items"


class TestCleanerPipeline:
    def test_chain_with_add(self):
        pipeline = HTMLCleaner() + WhitespaceCleaner()
        assert isinstance(pipeline, CleanerPipeline)
        result = pipeline.clean("<p>Hello    world</p>")
        assert result == "Hello world"

    def test_chain_multiple(self):
        pipeline = HTMLCleaner() + ControlCharRemover() + WhitespaceCleaner()
        assert isinstance(pipeline, CleanerPipeline)
        assert len(pipeline.cleaners) == 3

    def test_repr(self):
        pipeline = HTMLCleaner() + WhitespaceCleaner()
        assert "HTMLCleaner" in repr(pipeline)
        assert "WhitespaceCleaner" in repr(pipeline)
