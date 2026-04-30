import uuid

from fluiq.client import send_event
from fluiq.integrations.shared.chain import compute_chain_id

def log_trace(data):

    #adding metadata
    data['timestamp'] = __import__("time").time()

    if not data.get('trace_id'):
        data['trace_id'] = str(uuid.uuid4())

    if not data.get('parent_id'):
        chain_id = compute_chain_id(data)
        if chain_id is not None:
            data['parent_id'] = chain_id

    send_event(data)