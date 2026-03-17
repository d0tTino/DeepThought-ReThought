from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    action: str
    risk_level: str
    confidence_band: str
    reason: str
    artifacts: dict[str, Any]


class VersionedPolicyEngine:
    """Shared safety policy engine used by responder, selector, and gateway."""

    VERSION = "v1"

    _HIGH_RISK_TOKENS = {
        "kill",
        "bomb",
        "attack",
        "exploit",
        "harm",
        "suicide",
        "weapon",
        "malware",
    }
    _AMBIGUOUS_TOKENS = {
        "bypass",
        "hack",
        "password",
        "jailbreak",
        "leak",
        "phish",
    }

    def classify_input_risk(self, text: str) -> dict[str, Any]:
        lowered = (text or "").lower()
        tokens = {token for token in (self._HIGH_RISK_TOKENS | self._AMBIGUOUS_TOKENS) if token in lowered}
        if tokens & self._HIGH_RISK_TOKENS:
            level = "high"
        elif tokens & self._AMBIGUOUS_TOKENS:
            level = "ambiguous"
        else:
            level = "benign"
        return {
            "policy_version": self.VERSION,
            "stage": "pre_generation",
            "risk_level": level,
            "matched_tokens": sorted(tokens),
            "input_excerpt": (text or "")[:160],
        }

    def harden_prompt(self, user_input: str, *, risk_artifact: dict[str, Any]) -> dict[str, Any]:
        risk_level = str(risk_artifact.get("risk_level", "benign"))
        if risk_level == "high":
            hardening = "Refuse unsafe instructions and pivot to de-escalation guidance."
        elif risk_level == "ambiguous":
            hardening = "Ask clarifying questions before providing potentially sensitive guidance."
        else:
            hardening = "Provide concise, helpful, and policy-compliant responses."
        return {
            "policy_version": self.VERSION,
            "stage": "prompt_hardening",
            "risk_level": risk_level,
            "hardening": hardening,
            "hardened_prompt": f"{hardening} User request: {(user_input or '').strip()}",
        }

    def evaluate_candidate(self, *, text: str, confidence: float, prior_artifacts: list[dict[str, Any]] | None = None) -> PolicyDecision:
        prior = prior_artifacts or []
        prior_risk = next((a.get("risk_level") for a in prior if isinstance(a, dict) and a.get("stage") == "pre_generation"), "benign")
        classification = self.classify_input_risk(text)
        observed_risk = classification["risk_level"]
        risk_level = "high" if "high" in {prior_risk, observed_risk} else ("ambiguous" if "ambiguous" in {prior_risk, observed_risk} else "benign")
        confidence = max(0.0, min(1.0, float(confidence)))
        if confidence >= 0.8:
            band = "high"
        elif confidence >= 0.45:
            band = "medium"
        else:
            band = "low"

        if risk_level == "high":
            allowed = False
            action = "block"
            reason = "high_risk_content"
        elif risk_level == "ambiguous" and band == "high":
            allowed = False
            action = "escalate"
            reason = "ambiguous_high_confidence_requires_review"
        elif risk_level == "ambiguous":
            allowed = True
            action = "clarify"
            reason = "ambiguous_request"
        else:
            allowed = True
            action = "allow"
            reason = "benign"

        artifacts = {
            "policy_version": self.VERSION,
            "stage": "candidate_evaluation",
            "risk_level": risk_level,
            "confidence": confidence,
            "confidence_band": band,
            "action": action,
            "reason": reason,
            "matched_tokens": classification.get("matched_tokens", []),
            "prior_artifacts": prior,
        }
        return PolicyDecision(
            allowed=allowed,
            action=action,
            risk_level=risk_level,
            confidence_band=band,
            reason=reason,
            artifacts=artifacts,
        )

    def evaluate_egress(self, *, content: str, confidence: float | None, policy_artifacts: list[dict[str, Any]] | None = None) -> PolicyDecision:
        base = self.evaluate_candidate(text=content, confidence=float(confidence or 0.0), prior_artifacts=policy_artifacts)
        action = base.action
        allowed = base.allowed
        reason = base.reason
        if base.risk_level == "ambiguous" and base.confidence_band in {"medium", "high"}:
            allowed = False
            action = "escalate"
            reason = "egress_requires_escalation_for_ambiguous_confidence_band"

        artifacts = dict(base.artifacts)
        artifacts.update(
            {
                "stage": "egress",
                "action": action,
                "reason": reason,
                "egress_allowed": allowed,
            }
        )
        return PolicyDecision(
            allowed=allowed,
            action=action,
            risk_level=base.risk_level,
            confidence_band=base.confidence_band,
            reason=reason,
            artifacts=artifacts,
        )
