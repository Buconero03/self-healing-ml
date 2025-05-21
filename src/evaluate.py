def evaluate_model(predictions, ground_truth):
    from sklearn.metrics import f1_score
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    return f1_macro

def check_f1_threshold(f1_macro, threshold=0.999):
    if f1_macro < threshold:
        raise ValueError(f"F1-macro: {f1_macro} (threshold {threshold})")

if __name__ == "__main__":
# src/evaluate.py
import argparse, json, joblib, yaml, sys, numpy as np
from sklearn.metrics import f1_score

def load_jsonl(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line); X.append(row["x"]); y.append(row["y"])
    return np.array(X), np.array(y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/quality.yml")
    p.add_argument("--model",  default="model.joblib")
    args = p.parse_args()

    cfg   = yaml.safe_load(open(args.config))
    X_val, y_val = load_jsonl(cfg["data"]["val"])
    clf   = joblib.load(args.model)
    preds = clf.predict(X_val)
    f1    = f1_score(y_val, preds, average="macro")
    print(f"F1-macro: {f1:.4f} (threshold {cfg['metrics']['f1_macro']})")

    sys.exit(0 if f1 >= cfg["metrics"]["f1_macro"] else 1)

if __name__ == "__main__":
    main()

    pass
