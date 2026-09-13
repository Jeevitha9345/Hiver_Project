import pytest
from src.preprocessing import TextPreprocessor

def test_ascii_filter():
    assert TextPreprocessor.is_english_ascii("Where is my order? It was supposed to arrive today.") is True
    assert TextPreprocessor.is_english_ascii("¿Dónde está mi paquete?") is True  # Latin with accents is mostly ASCII
    assert TextPreprocessor.is_english_ascii("私の荷物はどこですか") is False  # Japanese kanji/hiragana

def test_clean_text_handles():
    raw = "@AmazonHelp @115712 where is my package?"
    cleaned = TextPreprocessor.clean_customer_query(raw)
    assert "@AmazonHelp" not in cleaned
    assert "where is my package?" in cleaned

def test_clean_text_urls():
    raw = "Check this tracking status: https://t.co/xyz12345 please!"
    cleaned = TextPreprocessor.clean_text(raw, replace_urls=True)
    assert "[URL]" in cleaned
    assert "https://t.co" not in cleaned
