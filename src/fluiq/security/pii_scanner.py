"""

PII detection and redaction using Microsoft Presidio.
Initialized once as a module-level singleton — never per-trace.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# ── Risk level ────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    CLEAN  = "clean"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# Weights per entity type (highest score wins for the trace)
_ENTITY_WEIGHTS: dict[str, float] = {
    "US_SSN":         1.0,
    "CREDIT_CARD":    1.0,
    "IBAN_CODE":      0.9,
    "CRYPTO":         0.8,
    "EMAIL_ADDRESS":  0.5,
    "PHONE_NUMBER":   0.5,
    "PERSON":         0.3,
    "IP_ADDRESS":     0.3,
    # custom API-key entities
    "OPENAI_API_KEY":     1.0,
    "ANTHROPIC_API_KEY":  1.0,
    "AWS_ACCESS_KEY":     1.0,
    "GITHUB_TOKEN":       1.0,
    "STRIPE_LIVE_KEY":    1.0,
}

_SUPPORTED_ENTITIES = list(_ENTITY_WEIGHTS.keys())


def _risk_from_score(score: float) -> RiskLevel:
    if score >= 0.9:
        return RiskLevel.HIGH
    if score >= 0.5:
        return RiskLevel.MEDIUM
    if score >= 0.3:
        return RiskLevel.LOW
    return RiskLevel.CLEAN


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PIIResult:
    detected:      bool
    entities:      List[str]
    risk_level:    RiskLevel
    redacted_text: str
    score:         float


# ── Custom recognizers ────────────────────────────────────────────────────────

def _build_custom_recognizers() -> list[PatternRecognizer]:
    specs = [
        (
            "OPENAI_API_KEY",
            [Pattern("OpenAI key", r"sk-[a-zA-Z0-9]{48}", 0.95)],
        ),
        (
            "ANTHROPIC_API_KEY",
            [Pattern("Anthropic key", r"sk-ant-[a-zA-Z0-9\-]{90,}", 0.95)],
        ),
        (
            "AWS_ACCESS_KEY",
            [Pattern("AWS key", r"AKIA[0-9A-Z]{16}", 0.95)],
        ),
        (
            "GITHUB_TOKEN",
            [Pattern("GitHub token", r"ghp_[a-zA-Z0-9]{36}", 0.95)],
        ),
        (
            "STRIPE_LIVE_KEY",
            [Pattern("Stripe live key", r"sk_live_[a-zA-Z0-9]{24}", 0.95)],
        ),
    ]
    return [
        PatternRecognizer(
            supported_entity=entity,
            patterns=patterns,
            supported_language="en",
        )
        for entity, patterns in specs
    ]


# ── Scanner ───────────────────────────────────────────────────────────────────

class FluiqPIIScanner:
    """Singleton-safe PII scanner. Instantiate once; call scan() repeatedly."""

    def __init__(self) -> None:
        self._analyzer  = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

        for recognizer in _build_custom_recognizers():
            self._analyzer.registry.add_recognizer(recognizer)

        logger.info("[fluiq.security] PII scanner ready")

    def scan(self, text: str) -> PIIResult:
        """Detect and redact PII in *text*. Never raises."""
        if not text or not text.strip():
            return PIIResult(
                detected=False,
                entities=[],
                risk_level=RiskLevel.CLEAN,
                redacted_text=text or "",
                score=0.0,
            )
        try:
            results = self._analyzer.analyze(
                text=text,
                entities=_SUPPORTED_ENTITIES,
                language="en",
            )

            if not results:
                return PIIResult(
                    detected=False,
                    entities=[],
                    risk_level=RiskLevel.CLEAN,
                    redacted_text=text,
                    score=0.0,
                )

            entity_types = list({r.entity_type for r in results})
            score = max(
                (_ENTITY_WEIGHTS.get(e, 0.3) for e in entity_types),
                default=0.0,
            )

            redacted = self._anonymizer.anonymize(
                text=text, analyzer_results=results
            ).text

            return PIIResult(
                detected=True,
                entities=entity_types,
                risk_level=_risk_from_score(score),
                redacted_text=redacted,
                score=score,
            )

        except Exception as exc:
            logger.exception("[fluiq.security] PII scan failed: %s", exc)
            return PIIResult(
                detected=False,
                entities=[],
                risk_level=RiskLevel.CLEAN,
                redacted_text=text,
                score=0.0,
            )