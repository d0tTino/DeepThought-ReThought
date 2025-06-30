from __future__ import annotations

import os

DB_PATH = os.getenv("SOCIAL_GRAPH_DB", "social_graph.db")
CURRENT_DB_PATH = DB_PATH

BULLYING_PHRASES = ["idiot", "stupid", "loser", "dumb", "ugly"]

MAX_MEMORY_LENGTH = 1000
MAX_THEORY_LENGTH = 256
MAX_PROMPT_LENGTH = 2000

# Idle/monitoring configuration
IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "5"))
PLAYFUL_REPLY_TIMEOUT_MINUTES = int(os.getenv("PLAYFUL_REPLY_TIMEOUT_MINUTES", "5"))
REFLECTION_CHECK_SECONDS = int(os.getenv("REFLECTION_CHECK_SECONDS", "300"))
BOT_CHAT_ENABLED = os.getenv("BOT_CHAT_ENABLED", "false").lower() in {
    "true",
    "1",
    "yes",
}

idle_response_candidates = [
    "Ever feel like everyone vanished?",
    "I'm still here if anyone wants to chat!",
    "Silence can be golden, but conversation is better.",
]

__all__ = [
    "DB_PATH",
    "CURRENT_DB_PATH",
    "BULLYING_PHRASES",
    "MAX_MEMORY_LENGTH",
    "MAX_THEORY_LENGTH",
    "MAX_PROMPT_LENGTH",
    "IDLE_TIMEOUT_MINUTES",
    "PLAYFUL_REPLY_TIMEOUT_MINUTES",
    "REFLECTION_CHECK_SECONDS",
    "BOT_CHAT_ENABLED",
    "idle_response_candidates",
]
