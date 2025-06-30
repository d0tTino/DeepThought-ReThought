from __future__ import annotations

import os

SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "textblob").lower()

if SENTIMENT_BACKEND == "vader":
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _sentiment = SentimentIntensityAnalyzer()

    def analyze_sentiment(text: str) -> float:
        """Return the compound sentiment score using VADER."""
        return _sentiment.polarity_scores(text)["compound"]

else:
    from textblob import TextBlob

    def analyze_sentiment(text: str) -> float:
        """Return the sentiment polarity using TextBlob."""
        return TextBlob(text).sentiment.polarity


__all__ = ["analyze_sentiment"]
