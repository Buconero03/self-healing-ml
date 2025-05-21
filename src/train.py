# src/train.py
import argparse, json, joblib, random, numpy as np, yaml, pathlib
from sklearn.linear_model import LogisticRegression

def load_jsonl(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            X.append(row["x"])
            y.append(row["y"])
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/quality.yml")
    parser.add_argument("--model-out", default="model.joblib")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    X_train, y_train = load_jsonl(cfg["data"]["train"])
    clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
    joblib.dump(clf, args.model_out)
    print(f"Model saved to {args.model_out}")

if __name__ == "__main__":
    main()

