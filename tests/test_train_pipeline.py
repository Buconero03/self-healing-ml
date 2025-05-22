"""
Smoke-test che addestra sul mini-dataset e verifica che:
1. venga creato il file modello
2. evaluate.py ritorni exit-code 0 ≥ soglia
"""
import subprocess, os, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG  = ROOT / "config" / "quality.yml"
MODEL_PATH = ROOT / "model.joblib"

def test_train_and_eval(tmp_path):
    # 1) train sul dataset mini
    rc = subprocess.call([
        "poetry", "run", "python", "src/train.py",
        "--config", str(CFG),
        "--model-out", str(MODEL_PATH)
    ])
    assert rc == 0 and MODEL_PATH.exists()

    # 2) evaluate: deve uscire 0
    rc = subprocess.call([
        "poetry", "run", "python", "src/evaluate.py",
        "--config", str(CFG),
        "--model",  str(MODEL_PATH)
    ])
    assert rc == 0

