class FluiqEvalError(Exception):
    """Raised by the SDK when ``fluiq.eval(mode='block')`` is active and one
    or more evaluation metrics score below their configured threshold.

    Attributes
    ----------
    failures : dict[str, float]
        ``{metric: score}`` for every metric that failed its threshold.
    scores : dict[str, float]
        All metric scores returned by the evaluation.
    """

    def __init__(
        self,
        failures: dict,
        scores: dict | None = None,
    ) -> None:
        failed_str = ", ".join(f"{m}={s:.3f}" for m, s in failures.items())
        super().__init__(f"Evaluation thresholds not met: {failed_str}")
        self.failures = failures
        self.scores   = scores or {}


class FluiqSecurityError(Exception):
    """Raised by the SDK when ``fluiq.secure(mode='block')`` is active and
    the pre-call check returns ``allow=False``.

    Attributes
    ----------
    risk_level : str
        One of ``"medium"`` or ``"high"``.
    attack_types : list[str]
        Which attack categories were detected (e.g. ``["jailbreak", "skeleton_key"]``).
    block_reason : str
        Human-readable explanation from the server.
    """

    def __init__(
        self,
        block_reason: str,
        risk_level: str = "high",
        attack_types: list[str] | None = None,
    ) -> None:
        super().__init__(block_reason)
        self.block_reason  = block_reason
        self.risk_level    = risk_level
        self.attack_types  = attack_types or []
