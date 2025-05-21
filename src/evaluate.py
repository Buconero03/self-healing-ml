```diff
import argparse
import json
import joblib
import yaml
import sys
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(predictions, ground_truth):
    \"\"\"
    Calcola F1-macro fra ground_truth e predictions.
    \"\"\"
    return f1_score(ground_truth, predictions, average='macro')


def check_f1_threshold(f1_macro, threshold=0.999):
    \"\"\"
    Solleva eccezione se f1_macro < threshold.
    \"\"\"
    if f1_macro < threshold:
        raise ValueError(f\"F1-macro: {f1_macro:.4f} (threshold {threshold})\")


def load_jsonl(path):
    \"\"\"
    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
    \"\"\"
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
            y.append(row['y'])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(\"--config\", default=\"config/quality.yml\", help=\"Percorso al file di configurazione YAML\")
    p.add_argument(\"--model\", default=\"model.joblib\", help=\"Percorso al modello serializzato\")
    args = p.parse_args()

import argparse
import json
import joblib
import yaml
import sys
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(predictions, ground_truth):
    \"\"\"
    Calcola F1-macro fra ground_truth e predictions.
    \"\"\"
    return f1_score(ground_truth, predictions, average='macro')


def check_f1_threshold(f1_macro, threshold=0.999):
    \"\"\"
    Solleva eccezione se f1_macro < threshold.
    \"\"\"
    if f1_macro < threshold:
        raise ValueError(f\"F1-macro: {f1_macro:.4f} (threshold {threshold})\")


def load_jsonl(path):
    \"\"\"
    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
    \"\"\"
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
            y.append(row['y'])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(\"--config\", default=\"config/quality.yml\", help=\"Percorso al file di configurazione YAML\")
    p.add_argument(\"--model\", default=\"model.joblib\", help=\"Percorso al modello serializzato\")
    args = p.parse_args()

import argparse
import json
import joblib
import yaml
import sys
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(predictions, ground_truth):
    \"\"\"
    Calcola F1-macro fra ground_truth e predictions.
    \"\"\"
    return f1_score(ground_truth, predictions, average='macro')


def check_f1_threshold(f1_macro, threshold=0.999):
    \"\"\"
    Solleva eccezione se f1_macro < threshold.
    \"\"\"
    if f1_macro < threshold:
        raise ValueError(f\"F1-macro: {f1_macro:.4f} (threshold {threshold})\")


def load_jsonl(path):
    \"\"\"
    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
    \"\"\"
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
            y.append(row['y'])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(\"--config\", default=\"config/quality.yml\", help=\"Percorso al file di configurazione YAML\")
    p.add_argument(\"--model\", default=\"model.joblib\", help=\"Percorso al modello serializzato\")
    args = p.parse_args()

import argparse
import json
import joblib
import yaml
import sys
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(predictions, ground_truth):
    \"\"\"
    Calcola F1-macro fra ground_truth e predictions.
    \"\"\"
    return f1_score(ground_truth, predictions, average='macro')


def check_f1_threshold(f1_macro, threshold=0.999):
    \"\"\"
    Solleva eccezione se f1_macro < threshold.
    \"\"\"
    if f1_macro < threshold:
        raise ValueError(f\"F1-macro: {f1_macro:.4f} (threshold {threshold})\")


def load_jsonl(path):
    \"\"\"
    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
    \"\"\"
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
            y.append(row['y'])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(\"--config\", default=\"config/quality.yml\", help=\"Percorso al file di configurazione YAML\")
    p.add_argument(\"--model\", default=\"model.joblib\", help=\"Percorso al modello serializzato\")
    args = p.parse_args()

import argparse
import json
import joblib
import yaml
import sys
import numpy as np
from sklearn.metrics import f1_score


def evaluate_model(predictions, ground_truth):
    \"\"\"
    Calcola F1-macro fra ground_truth e predictions.
    \"\"\"
    return f1_score(ground_truth, predictions, average='macro')


def check_f1_threshold(f1_macro, threshold=0.999):
    \"\"\"
    Solleva eccezione se f1_macro < threshold.
    \"\"\"
    if f1_macro < threshold:
        raise ValueError(f\"F1-macro: {f1_macro:.4f} (threshold {threshold})\")


def load_jsonl(path):
    \"\"\"
    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
    \"\"\"
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            row = json.loads(line)
            X.append(row['x'])
            y.append(row['y'])
    return np.array(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(\"--config\", default=\"config/quality.yml\", help=\"Percorso al file di configurazione YAML\")
    p.add_argument(\"--model\", default=\"model.joblib\", help=\"Percorso al modello serializzato\")
    args = p.parse_args()

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


def check_f1_threshold(f1_macro, threshold=0.999):
    """
    Solleva eccezione se f1_macro < threshold.
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
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/quality.yml", help="Percorso al file di configurazione YAML")
    p.add_argument("--model", default="model.joblib", help="Percorso al modello serializzato")
    args = p.parse_args()

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


def check_f1_threshold(f1_macro, threshold=0.999):
    """
    Solleva eccezione se f1_macro < threshold.
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
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/quality.yml", help="Percorso al file di configurazione YAML")
    p.add_argument("--model", default="model.joblib", help="Percorso al modello serializzato")
    args = p.parse_args()

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


def check_f1_threshold(f1_macro, threshold=0.999):
    """
    Solleva eccezione se f1_macro < threshold.
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
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/quality.yml", help="Percorso al file di configurazione YAML")
    p.add_argument("--model", default="model.joblib", help="Percorso al modello serializzato")
    args = p.parse_args()

diff --git a/src/evaluate.py b/src/evaluate.py
index 1c2d3e4..5f6g7h8 100644
--- a/src/evaluate.py
+++ b/src/evaluate.py
@@ -1,0 +1,43 @@
+import argparse
+import json
+import joblib
+import yaml
+import sys
+import numpy as np
+from sklearn.metrics import f1_score
+
+
+def evaluate_model(predictions, ground_truth):
+    """
+    Calcola F1-macro fra ground_truth e predictions.
+    """
+    return f1_score(ground_truth, predictions, average='macro')
+
+
+def check_f1_threshold(f1_macro, threshold=0.999):
+    """
+    Solleva eccezione se f1_macro < threshold.
+    """
+    if f1_macro < threshold:
+        raise ValueError(f"F1-macro: {f1_macro:.4f} (threshold {threshold})")
+
+
+def load_jsonl(path):
+    """
+    Legge un file JSONL con campi 'x' e 'y', restituisce (X, y) come numpy array.
+    """
+    X, y = [], []
+    with open(path, 'r') as f:
+        for line in f:
+            row = json.loads(line)
+            X.append(row['x'])
+            y.append(row['y'])
+    return np.array(X), np.array(y)
+
+
+def main():
+    p = argparse.ArgumentParser()
+    p.add_argument("--config", default="config/quality.yml", help="Percorso al file di configurazione YAML")
+    p.add_argument("--model", default="model.joblib", help="Percorso al modello serializzato")
+    args = p.parse_args()
+
+    # Carica configurazione e dati
+    cfg = yaml.safe_load(open(args.config, 'r'))
+    X_val, y_val = load_jsonl(cfg["data"]["val"])
+
+    # Carica modello e fai predizioni
+    clf = joblib.load(args.model)
+    preds = clf.predict(X_val)
+
+    # Calcola F1-macro e confronta col threshold
+    f1 = f1_score(y_val, preds, average="macro")
+    print(f"F1-macro: {f1:.4f} (threshold {cfg['metrics']['f1_macro']})")
+
+    # Esci con 0 se supera soglia, altrimenti 1
+    sys.exit(0 if f1 >= cfg['metrics']['f1_macro'] else 1)
+
+
+if __name__ == "__main__":
+    main()
```