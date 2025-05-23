import json
import sys
import time

"""
Helper per il logging strutturato in formato JSON.
Ogni record viene stampato su stderr e può essere ingerito da Grafana Loki / ELK.
"""

def log(event: str, **fields):
    """
    Stampa un record JSON con timestamp e campi aggiuntivi.

    :param event: nome dell'evento (es. "train_start", "patch_apply").
    :param fields: campi addizionali da includere nel record.
    """
    record = {
        "ts": time.time(),
        "event": event
    }
    record.update(fields)
    print(json.dumps(record), file=sys.stderr, flush=True)

