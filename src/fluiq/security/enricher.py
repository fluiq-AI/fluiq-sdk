"""fluiq/security/enricher.py

SecurityEnricher runs all three scanners on a trace dict and writes
security fields back into it. Singletons are created once at import time.

Never raises — if scanning fails the trace is forwarded unchanged.
Never stores the original prompt/response when risk is HIGH.
"""
from __future__ import annotations

import logging
from typing import Any

from .injection_scanner import FluiqInjectionScanner, InjectionResult
from .pii_scanner import FluiqPIIScanner, PIIResult, RiskLevel
from .secret_scanner import FluiqSecretScanner, SecretResult

logger = logging.getLogger(__name__)

# ── Module-level singletons (initialised once) ────────────────────────────────
_pii_scanner       = FluiqPIIScanner()
_injection_scanner = FluiqInjectionScanner()
_secret_scanner    = FluiqSecretScanner()

# Risk level ordering for max() comparisons
_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.CLEAN:  0,
    RiskLevel.LOW:    1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH:   3,
}


def _max_risk(*levels: RiskLevel) -> RiskLevel:
    return max(levels, key=lambda r: _RISK_ORDER[r])


def _extract_text(value: Any) -> str:
    """Best-effort text extraction from prompt/response fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("text") or ""
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return str(value)


class SecurityEnricher:
    """Enrich a raw trace dict with security scan results."""

    def enrich(self, trace: dict[str, Any]) -> dict[str, Any]:
        """
        Run all scanners and write security fields into *trace*.
        Returns the (mutated) trace dict. Never raises.
        """
        try:
            prompt_text   = _extract_text(trace.get("input") or trace.get("messages"))
            response_text = _extract_text(trace.get("response") or trace.get("output"))

            # ── Scan ──────────────────────────────────────────────────────────
            prompt_pii:       PIIResult       = _pii_scanner.scan(prompt_text)
            response_pii:     PIIResult       = _pii_scanner.scan(response_text)
            prompt_injection: InjectionResult = _injection_scanner.scan(prompt_text)
            response_secrets: SecretResult    = _secret_scanner.scan(response_text)

            # ── Aggregate risk ────────────────────────────────────────────────
            overall_risk = _max_risk(
                prompt_pii.risk_level,
                response_pii.risk_level,
                prompt_injection.risk_level,
                response_secrets.risk_level,
            )
            risk_score = max(
                prompt_pii.score,
                response_pii.score,
                prompt_injection.risk_score,
                1.0 if response_secrets.detected else 0.0,
            )

            # ── Redacted copies ───────────────────────────────────────────────
            prompt_redacted   = prompt_pii.redacted_text
            response_redacted = response_pii.redacted_text

            # For HIGH risk: replace originals with redacted versions so we
            # never persist sensitive data.
            if overall_risk == RiskLevel.HIGH:
                if "input" in trace and isinstance(trace["input"], str):
                    trace["input"] = prompt_redacted
                if "messages" in trace:
                    trace["messages"] = None  # too complex to redact in-place
                if "response" in trace and isinstance(trace["response"], str):
                    trace["response"] = response_redacted
                if "output" in trace and isinstance(trace["output"], str):
                    trace["output"] = response_redacted

            # ── Write security fields ─────────────────────────────────────────
            trace.update({
                "prompt_redacted":       prompt_redacted,
                "response_redacted":     response_redacted,
                "security_risk_level":   overall_risk.value,
                "security_risk_score":   round(risk_score, 4),
                "pii_entities_prompt":   prompt_pii.entities,
                "pii_entities_response": response_pii.entities,
                "injection_detected":    prompt_injection.detected,
                "injection_patterns":    prompt_injection.patterns_found,
                "secrets_detected":      response_secrets.detected,
                "secret_types":          response_secrets.secret_types,
            })

        except Exception as exc:
            logger.exception("[fluiq.security] enrichment failed: %s", exc)
            # Fail open — forward the trace unchanged rather than dropping it

        return trace