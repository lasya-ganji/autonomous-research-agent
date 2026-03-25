import time
import hashlib
from datetime import datetime


def hash_data(data):
    try:
        return hashlib.md5(str(data).encode()).hexdigest()
    except Exception:
        return "hash_error"


def log_node_execution(node_name: str, input_data, output_data, start_time: float):
    duration = int((time.time() - start_time) * 1000)

    log = {
        "node": node_name,
        "input_hash": hash_data(input_data),
        "output_hash": hash_data(output_data),
        "duration_ms": duration,
        "timestamp": datetime.now().astimezone().isoformat()
    }

    print("[LOG]", log)