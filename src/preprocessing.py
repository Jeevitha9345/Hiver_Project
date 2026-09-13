import re
import html
from typing import Optional

class TextPreprocessor:
    """
    Careful text preprocessor for Twitter customer-support interactions.
    Preserves informal sentiment cues, punctuation, and domain terms
    while sanitizing handles, URLs, and encoding artifacts.
    """
    
    HANDLE_PATTERN = re.compile(r'@[A-Za-z0-9_]+')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MULTIPLE_SPACES = re.compile(r'\s+')
    
    @staticmethod
    def is_english_ascii(text: str, threshold: float = 0.85) -> bool:
        """Checks whether the text is predominantly ASCII (filters foreign alphabets/scripts)."""
        if not text or not isinstance(text, str):
            return False
        clean = text.strip()
        if len(clean) == 0:
            return False
        ascii_count = sum(1 for c in clean if ord(c) < 128)
        return (ascii_count / len(clean)) >= threshold

    @classmethod
    def clean_text(cls, text: str, replace_urls: bool = True, normalize_handles: bool = True) -> str:
        """
        Normalizes a tweet text:
        1. Unescapes HTML entities (&amp; -> &)
        2. Normalizes URLs to [URL] or removes them
        3. Normalizes user mentions to @user / @brand
        4. Cleans whitespace while preserving casing and sentiment punctuation
        """
        if not isinstance(text, str):
            return ""
        
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Replace URLs
        if replace_urls:
            text = cls.URL_PATTERN.sub('[URL]', text)
        else:
            text = cls.URL_PATTERN.sub('', text)
            
        # Normalize handles
        if normalize_handles:
            # Replace target brand mention with @brand
            text = re.sub(r'(?i)@amazonhelp\b', '@brand', text)
            # Other numeric/anonymized customer handles to @user
            text = cls.HANDLE_PATTERN.sub('@user', text)
            
        # Collapse multiple spaces and trim
        text = cls.MULTIPLE_SPACES.sub(' ', text).strip()
        return text

    @classmethod
    def clean_customer_query(cls, text: str) -> str:
        """Specialized cleaning for incoming customer query."""
        cleaned = cls.clean_text(text, replace_urls=True, normalize_handles=True)
        # Strip leading @user or @brand mentions from the beginning of customer query
        cleaned = re.sub(r'^(@(user|brand)\s*)+', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    @classmethod
    def clean_support_reply(cls, text: str) -> str:
        """Specialized cleaning for historical support responses."""
        cleaned = cls.clean_text(text, replace_urls=False, normalize_handles=True)
        # Strip leading customer mentions
        cleaned = re.sub(r'^(@(user|brand)\s*)+', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned
