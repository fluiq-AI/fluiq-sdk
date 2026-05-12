import uuid

from fluiq.client import send_event
from fluiq.integrations.shared.chain import compute_chain_id
from fluiq.integrations.shared.context import current_llm_trace_id, current_parent_id

from fluiq.config import _config

_enricher = None

def _get_enricher():
    global _enricher
    if _enricher is None:
        from fluiq.security.enricher import SecurityEnricher
        _enricher = SecurityEnricher()
    return _enricher

def log_trace(data):
    try:
        data['timestamp'] = __import__("time").time()

        if not data.get('trace_id'):
            ctx_trace_id = current_llm_trace_id()
            data['trace_id'] = ctx_trace_id or str(uuid.uuid4())

        if not data.get('parent_id'):
            ctx_parent = current_parent_id()
            if ctx_parent:
                data['parent_id'] = ctx_parent
            else:
                chain_id = compute_chain_id(data)
                if chain_id is not None:
                    data['parent_id'] = chain_id

        global_enabled = _config.get("security_scan", True)
        per_trace_enabled = data.pop("_security_scan", True)

        if global_enabled and per_trace_enabled:
            try:
                _get_enricher().enrich(data)
            except Exception:
                pass
            
        send_event(data)
    except Exception:
        pass