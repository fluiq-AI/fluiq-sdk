"""fluiq/security/secret_scanner.py

Detects leaked credentials and high-entropy strings in LLM outputs.
Initialized once as a module-level singleton.
"""
from __future__ import annotations

import logging
import math
import re
import string
from dataclasses import dataclass, field
from typing import List

from .pii_scanner import RiskLevel

logger = logging.getLogger(__name__)

# ── Regex patterns for known secret formats ───────────────────────────────────

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key",     re.compile(r"sk-[a-zA-Z0-9]{48}")),
    ("anthropic_key",  re.compile(r"sk-ant-[a-zA-Z0-9\-]{90,}")),
    ("aws_key",        re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token",   re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("stripe_live_key", re.compile(r"sk_live_[a-zA-Z0-9]{24}")),
]

# Entropy threshold — strings above this are flagged as potential secrets
_ENTROPY_THRESHOLD = 4.5

# Minimum token length to entropy-check (short tokens are noise)
_MIN_ENTROPY_TOKEN_LEN = 20


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SecretResult:
    detected:              bool
    secret_types:          List[str]
    high_entropy_detected: bool
    risk_level:            RiskLevel


# ── Entropy helpers ───────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy (bits) for string *s*."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")


def _high_entropy_tokens(text: str) -> bool:
    """Return True if any token in *text* exceeds the entropy threshold."""
    for match in _TOKEN_RE.finditer(text):
        token = match.group()
        if len(token) >= _MIN_ENTROPY_TOKEN_LEN:
            if _shannon_entropy(token) > _ENTROPY_THRESHOLD:
                return True
    return False


# ── Scanner ───────────────────────────────────────────────────────────────────

class FluiqSecretScanner:
    """Singleton-safe credential/secret scanner."""

    def scan(self, text: str) -> SecretResult:
        """Scan *text* for secrets and high-entropy strings. Never raises."""
        if not text or not text.strip():
            return SecretResult(
                detected=False,
                secret_types=[],
                high_entropy_detected=False,
                risk_level=RiskLevel.CLEAN,
            )
        try:
            found_types: list[str] = [
                label
                for label, pattern in _SECRET_PATTERNS
                if pattern.search(text)
            ]

            high_entropy = _high_entropy_tokens(text)

            detected = bool(found_types) or high_entropy

            if not detected:
                return SecretResult(
                    detected=False,
                    secret_types=[],
                    high_entropy_detected=False,
                    risk_level=RiskLevel.CLEAN,
                )

            # Known secret patterns → HIGH; entropy-only → MEDIUM
            risk_level = RiskLevel.HIGH if found_types else RiskLevel.MEDIUM

            return SecretResult(
                detected=True,
                secret_types=found_types,
                high_entropy_detected=high_entropy,
                risk_level=risk_level,
            )

        except Exception as exc:
            logger.exception("[fluiq.security] secret scan failed: %s", exc)
            return SecretResult(
                detected=False,
                secret_types=[],
                high_entropy_detected=False,
                risk_level=RiskLevel.CLEAN,
            )