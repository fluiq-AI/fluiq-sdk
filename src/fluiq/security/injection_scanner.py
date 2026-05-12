"""fluiq/security/injection_scanner.py

Prompt injection detection using regex and substring matching.
Initialized once as a module-level singleton.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from .pii_scanner import RiskLevel

logger = logging.getLogger(__name__)

# ── Injection patterns ────────────────────────────────────────────────────────

_RAW_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your",
    "you are now",
    "pretend you are",
    "act as if",
    "jailbreak",
    "bypass",
    "override system",
    "forget everything",
    "new persona",
    "do anything now",
    "DAN",
    "hypothetically",
    "for educational purposes",
]

# Pre-compile: case-insensitive word/phrase match
_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (p, re.compile(re.escape(p), re.IGNORECASE))
    for p in _RAW_PATTERNS
]

_TOTAL = len(_RAW_PATTERNS)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InjectionResult:
    detected:       bool
    patterns_found: List[str]
    risk_score:     float
    risk_level:     RiskLevel


# ── Scanner ───────────────────────────────────────────────────────────────────

class FluiqInjectionScanner:
    """Singleton-safe prompt-injection scanner."""

    def scan(self, text: str) -> InjectionResult:
        """Scan *text* for prompt injection patterns. Never raises."""
        if not text or not text.strip():
            return InjectionResult(
                detected=False,
                patterns_found=[],
                risk_score=0.0,
                risk_level=RiskLevel.CLEAN,
            )
        try:
            found = [
                pattern_str
                for pattern_str, compiled in _COMPILED
                if compiled.search(text)
            ]

            if not found:
                return InjectionResult(
                    detected=False,
                    patterns_found=[],
                    risk_score=0.0,
                    risk_level=RiskLevel.CLEAN,
                )

            risk_score = len(found) / _TOTAL
            # Any detection is at minimum MEDIUM
            if risk_score < 0.5:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 0.9:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.HIGH

            return InjectionResult(
                detected=True,
                patterns_found=found,
                risk_score=round(risk_score, 4),
                risk_level=risk_level,
            )

        except Exception as exc:
            logger.exception("[fluiq.security] injection scan failed: %s", exc)
            return InjectionResult(
                detected=False,
                patterns_found=[],
                risk_score=0.0,
                risk_level=RiskLevel.CLEAN,
            )