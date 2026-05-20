"""fluiq.evals — server-side evaluation via fluiq.eval().

All evaluation runs on the Fluiq backend (LLM-as-judge) so no vendor
API keys are required in the SDK.  Call fluiq.eval() after
fluiq.instrument() to activate.

Warn mode: SDK embeds _eval_config in the trace → /ingest fans out to worker.
Block mode: SDK calls /evaluate synchronously via call_evaluate_block().
"""
from fluiq.evals.client import call_evaluate_block

__all__ = ["call_evaluate_block"]