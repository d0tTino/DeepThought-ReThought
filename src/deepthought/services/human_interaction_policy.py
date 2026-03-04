from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionPolicyDecision:
    delay_seconds: float
    typing_seconds: float
    cooldown_seconds: float
    style_modifiers: list[str] = field(default_factory=list)


class HumanInteractionPolicy:
    """Compute human-like timing and style hints for outbound Discord responses."""

    def __init__(
        self,
        *,
        min_delay_seconds: float = 0.2,
        max_delay_seconds: float = 4.0,
        typing_chars_per_second: float = 18.0,
        max_typing_seconds: float = 6.0,
    ) -> None:
        self._min_delay = max(0.0, min_delay_seconds)
        self._max_delay = max(self._min_delay, max_delay_seconds)
        self._typing_cps = max(1.0, typing_chars_per_second)
        self._max_typing = max(0.0, max_typing_seconds)

    def decide(
        self,
        *,
        message_text: str,
        channel_pace: float,
        familiarity: float,
        metadata: dict[str, object] | None = None,
    ) -> InteractionPolicyDecision:
        """Return timing and style decisions.

        Args:
            message_text: message to be sent.
            channel_pace: estimated messages/minute in channel.
            familiarity: [0, 1] estimate for how familiar this user/channel is.
            metadata: optional planner/selector hints that override computed values.
        """

        text = message_text or ""
        length_factor = min(len(text) / 240.0, 1.0)
        pace_factor = min(max(channel_pace, 0.0) / 30.0, 1.0)
        familiarity_factor = min(max(familiarity, 0.0), 1.0)

        baseline = self._min_delay + (self._max_delay - self._min_delay) * (0.45 * length_factor + 0.35 * pace_factor)
        baseline *= 1.15 - (0.45 * familiarity_factor)
        delay_seconds = min(max(baseline, self._min_delay), self._max_delay)

        typing_seconds = min(len(text) / self._typing_cps, self._max_typing)
        cooldown_seconds = min(3.0, 0.25 + 0.75 * length_factor + 0.5 * pace_factor)

        style_modifiers: list[str] = []
        if pace_factor > 0.65:
            style_modifiers.append("concise")
        if familiarity_factor > 0.70:
            style_modifiers.append("warm")
        elif familiarity_factor < 0.25:
            style_modifiers.append("formal")
        if len(text) > 300:
            style_modifiers.append("structured")

        if metadata:
            raw_delay = metadata.get("delay_seconds")
            if isinstance(raw_delay, (int, float)):
                delay_seconds = min(max(float(raw_delay), 0.0), self._max_delay)
            raw_typing = metadata.get("typing_seconds")
            if isinstance(raw_typing, (int, float)):
                typing_seconds = min(max(float(raw_typing), 0.0), self._max_typing)
            raw_cooldown = metadata.get("cooldown_seconds")
            if isinstance(raw_cooldown, (int, float)):
                cooldown_seconds = min(max(float(raw_cooldown), 0.0), 10.0)
            raw_modifiers = metadata.get("style_modifiers")
            if isinstance(raw_modifiers, list):
                style_modifiers = [m for m in raw_modifiers if isinstance(m, str)]

        return InteractionPolicyDecision(
            delay_seconds=round(delay_seconds, 3),
            typing_seconds=round(typing_seconds, 3),
            cooldown_seconds=round(cooldown_seconds, 3),
            style_modifiers=style_modifiers,
        )
