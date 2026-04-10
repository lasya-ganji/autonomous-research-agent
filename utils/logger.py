import hashlib
from datetime import datetime


def hash_data(data):
    try:
        return hashlib.md5(str(data).encode()).hexdigest()
    except Exception:
        return "hash_error"


def log_node_execution(node_name: str, input_data=None, output_data=None):

    log = {
        "node": node_name,
        "input_hash": hash_data(input_data),
        "output_hash": hash_data(output_data),
        "timestamp": datetime.now().astimezone().isoformat()
    }

    print("[LOG]", log)