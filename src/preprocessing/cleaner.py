"""
Raw Text Sanitization: Modular preprocessing using re and NLTK.
Strips HTML tags, URLs, emojis, punctuation, and linguistic noise.
"""
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextCleaner:
    """Modular text preprocessing with lemmatization and noise removal."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text: str) -> str:
        """Full cleaning pipeline: strip HTML, URLs, emojis, punctuation, lowercase, lemmatize."""
        if not text or pd.isna(text):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)       # Remove URLs
        text = re.sub(r'<[^>]+>', '', text)                        # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)                    # Remove punctuation, digits, emojis
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word.lower())
                  for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(tokens)

    def tokenize(self, text: str) -> list:
        """Tokenize and lemmatize without removing stopwords (for highlighting)."""
        if not text or pd.isna(text):
            return []
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        return [self.lemmatizer.lemmatize(word.lower()) for word in tokens]