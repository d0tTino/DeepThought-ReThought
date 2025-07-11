import logging
from typing import Optional

from textblob import TextBlob

from .user_graph_dal import UserGraphDAL

logger = logging.getLogger(__name__)


class SocialGraphMemory:
    """Record messages and sentiment in a :class:`UserGraphDAL`."""

    def __init__(self, dal: Optional[UserGraphDAL] = None) -> None:
        self._dal = dal or UserGraphDAL()

    def record_message(self, source: str, text: str, target: Optional[str] = None) -> None:
        """Analyze sentiment of ``text`` and store the interaction."""
        try:
            score = float(TextBlob(text).sentiment.polarity)
        except Exception:  # pragma: no cover - TextBlob failure
            logger.exception("Sentiment analysis failed")
            score = 0.0
        self._dal.add_message(source, target, sentiment_score=score)

    # Expose some helper methods from the DAL
    def get_affinity(self, user_id: str) -> int:
        return self._dal.get_affinity(user_id)

    def get_friendliness(self, source: str, target: str) -> float:
        return self._dal.get_friendliness(source, target)

    def get_hostility(self, source: str, target: str) -> float:
        return self._dal.get_hostility(source, target)
