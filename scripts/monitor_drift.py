#!/usr/bin/env python3
"""
Monitor di drift dei dati.
Confronta la distribuzione attuale del dataset di validazione con quella di riferimento
usando Kolmogorov-Smirnov test per ogni feature numerica.
Se uno dei test fallisce (p < 0.05), esce con codice 1.
"""
import json
import sys
import argparse
import numpy as np
from scipy.stats import ks_2samp

def load_jsonl(path):
    X = []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
    return np.array(X)

def main():
    p = argparse.ArgumentParser(description="Monitor drift dei dati")
    p.add_argument('--baseline', required=True,
                   help='Percorso al JSONL di baseline (train)')
    p.add_argument('--current', required=True,
                   help='Percorso al JSONL corrente (val)')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Soglia di significatività per KS-test')
    args = p.parse_args()

    # Carica dati
    X_base = load_jsonl(args.baseline)
    X_cur  = load_jsonl(args.current)

    # Assumiamo X.shape = (N_samples, N_features)
    n_features = X_base.shape[1]
    failed = []
    for j in range(n_features):
        stat, pval = ks_2samp(X_base[:, j], X_cur[:, j])
        if pval < args.alpha:
            failed.append((j, pval))
    
    if failed:
        print(json.dumps({
            'event': 'drift_detected',
            'failed_features': [f[0] for f in failed],
            'pvalues': [f[1] for f in failed]
        }))
        sys.exit(1)
    else:
        print(json.dumps({'event': 'no_drift'}))
        sys.exit(0)

if __name__ == '__main__':
    main()

