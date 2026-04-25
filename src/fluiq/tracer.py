from fluiq.client import send_event

def log_trace(data):

    #adding metadata
    data['timestamp'] = __import__("time").time()

    send_event(data)