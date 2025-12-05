import json
from datetime import datetime

def log_to_file(prompt, response, metadata):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
        **metadata
    }

    with open("llm_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")