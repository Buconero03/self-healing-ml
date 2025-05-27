#!/usr/bin/env python3
import argparse
import json
import joblib
import yaml
import sys
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(predictions, ground_truth):
    """
    Calcola F1-macro fra ground_truth e predictions.
    """
    return f1_score(ground_truth, predictions, average='macro')


def check_f1_threshold(f1_macro, threshold):
    """
    Solleva ValueError se f1_macro < threshold.
    """
    if f1_macro < threshold:
        raise ValueError(f"F1-macro: {f1_macro:.4f} (threshold {threshold})")


def load_jsonl(path):
    """
    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
    """
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
            y.append(row['y'])
    return np.array(X), np.array(y)


def main():
    # 1) Parsing degli argomenti
    p = argparse.ArgumentParser(
        description="Valuta un modello e verifica soglia F1-macro."
    )
    p.add_argument(
        "--config", default="config/quality.yml",
        help="Percorso al file di configurazione YAML"
    )
    p.add_argument(
        "--model", default="model.joblib",
        help="Percorso al modello serializzato"
    )
    args = p.parse_args()

    # 2) Carica configurazione e dati
    cfg      = yaml.safe_load(open(args.config, 'r'))
    X_val, y_val = load_jsonl(cfg["data"]["val"])

    # 3) Carica modello e fai predizioni
    clf   = joblib.load(args.model)
    preds = clf.predict(X_val)

    # DEBUG: stampa subito il valore raw di F1
    print(f"DEBUG: F1-macro = {f1_score(y_val, preds, average='macro'):.4f}")

    # 4) Calcola F1-macro
    f1 = evaluate_model(preds, y_val)
    print(f"F1-macro: {f1:.4f} (threshold {cfg['metrics']['f1_macro']})")

    # 5) Verifica soglia: solleva ValueError se sotto threshold
    check_f1_threshold(f1, cfg['metrics']['f1_macro'])

    # 6) Se siamo qui, tutto OK → exit code 0
    sys.exit(0)


if __name__ == "__main__":
    main()
