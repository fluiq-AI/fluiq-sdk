from typing import Any, List, Optional

from fluiq.evaluations.base import (
    BaseEvaluator,
    EvalResult,
    LLMJudge,
    _clamp_unit,
    _coerce_contexts,
)


_CLAIMS_PROMPT = (
    "Extract every standalone factual claim from the ANSWER below. "
    "Return JSON: {{\"claims\": [\"claim 1\", \"claim 2\", ...]}}.\n\n"
    "ANSWER:\n{answer}"
)


_VERIFY_PROMPT = (
    "You are checking whether each CLAIM is supported by the REFERENCE. "
    "A claim is SUPPORTED only if the reference entails it; if the reference "
    "neither states nor implies it, mark it UNSUPPORTED. Speculation, added "
    "details, and contradictions are UNSUPPORTED.\n\n"
    "REFERENCE:\n{reference}\n\n"
    "CLAIMS:\n{claims}\n\n"
    "Return JSON: {{\"verdicts\": [{{\"claim\": str, \"supported\": bool, "
    "\"reason\": str}}]}}"
)


class HallucinationEvaluator(BaseEvaluator):
    """Hallucination detector.

    Score is 1.0 when no hallucinated claims are detected and 0.0 when every
    claim in the answer is unsupported. Lower scores are worse. The evaluator
    accepts either retrieved `contexts` (RAG style) or a single `reference`
    string (closed-book / ground truth) as the source of truth.
    """

    name = "hallucination"

    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        threshold: float = 0.7,
    ):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(
        self,
        answer: str,
        contexts: Any = None,
        reference: Optional[str] = None,
        **_: Any,
    ) -> EvalResult:
        if not answer or not answer.strip():
            return self._result(
                score=1.0,
                reason="empty answer; nothing to hallucinate",
                details={"claims": [], "verdicts": []},
            )

        reference_text = _build_reference(contexts, reference)
        if not reference_text:
            return self._result(
                score=0.0,
                reason="no reference or contexts provided",
                details={"claims": [], "verdicts": []},
            )

        claims = self._extract_claims(answer)
        if not claims:
            return self._result(
                score=1.0,
                reason="no factual claims extracted from answer",
                details={"claims": [], "verdicts": []},
            )

        verdicts = self._verify_claims(reference_text, claims)
        if not verdicts:
            return self._result(
                score=0.0,
                reason="judge returned no verdicts",
                details={"claims": claims, "verdicts": []},
            )

        supported = sum(1 for v in verdicts if v.get("supported") is True)
        score = _clamp_unit(supported / len(verdicts))
        unsupported = [v for v in verdicts if v.get("supported") is not True]
        reason = (
            f"{supported}/{len(verdicts)} claims supported"
            if not unsupported
            else f"{len(unsupported)} unsupported claim(s) of {len(verdicts)}"
        )
        return self._result(
            score=score,
            reason=reason,
            details={
                "claims": claims,
                "verdicts": verdicts,
                "supported_count": supported,
                "total_claims": len(verdicts),
            },
        )

    def _extract_claims(self, answer: str) -> List[str]:
        data = self.judge.judge_json(_CLAIMS_PROMPT.format(answer=answer))
        raw = data.get("claims") or []
        if not isinstance(raw, list):
            return []
        return [str(c).strip() for c in raw if str(c).strip()]

    def _verify_claims(self, reference: str, claims: List[str]) -> List[dict]:
        prompt = _VERIFY_PROMPT.format(
            reference=reference,
            claims="\n".join(f"- {c}" for c in claims),
        )
        data = self.judge.judge_json(prompt)
        raw = data.get("verdicts") or []
        if not isinstance(raw, list):
            return []
        verdicts: List[dict] = []
        for v in raw:
            if not isinstance(v, dict):
                continue
            verdicts.append({
                "claim": str(v.get("claim", "")).strip(),
                "supported": bool(v.get("supported")),
                "reason": str(v.get("reason", "")).strip() or None,
            })
        return verdicts


def _build_reference(contexts: Any, reference: Optional[str]) -> str:
    parts: List[str] = []
    for c in _coerce_contexts(contexts):
        if c.strip():
            parts.append(c.strip())
    if reference and reference.strip():
        parts.append(reference.strip())
    return "\n\n".join(parts)
