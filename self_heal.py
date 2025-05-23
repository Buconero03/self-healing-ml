#!/usr/bin/env python3
"""Self-healing orchestrator
----------------------------------------
Allena `train.py` e valuta `evaluate.py`. Se la valutazione fallisce,
richiede all'LLM una patch diff; in caso di hunk falliti,
esegue un fallback full-file. Registra eventi chiave col JSON logger.
"""

import os
import subprocess
import json
import sys
import re
import time
from pathlib import Path
import yaml
from openai import OpenAI
from src.logger import log

# ─── Config ───────────────────────────────────────────────────
CONFIG_PATH = Path("config/quality.yml")
if not CONFIG_PATH.exists():
    sys.exit(f"Config file not found: {CONFIG_PATH}")

cfg = yaml.safe_load(CONFIG_PATH.read_text())
MAX_ITERS       = cfg.get("self_heal", {}).get("max_iterations", 5)
MODEL           = cfg.get("model_name", "model.joblib")
TRAIN_TIMEOUT   = 300    # seconds
MAX_PATCH_BYTES = 500    # bytes

# ─── OpenAI client ────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY missing.")
client = OpenAI(api_key=api_key)

# ─── Shell helper ──────────────────────────────────────────────
def run(cmd: list[str], timeout: int | None = None):
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr

# ─── Train / Evaluate ─────────────────────────────────────────
def train():
    return run([
        "poetry", "run", "python", "src/train.py",
        "--config", str(CONFIG_PATH), "--model-out", MODEL
    ], timeout=TRAIN_TIMEOUT)

def evaluate():
    return run([
        "poetry", "run", "python", "src/evaluate.py",
        "--config", str(CONFIG_PATH), "--model", MODEL
    ])

# ─── Prompt helpers ────────────────────────────────────────────
def SYSTEM_PROMPT() -> str:
    return (
        "Sei un assistente che corregge bug in progetti Python seguendo PEP8 e senza introdurre nuove dipendenze. "
        "Quando proponi una patch, restituisci SOLO il diff unified (git diff -U0) con percorsi relativi e intestazioni complete. "
        "Il diff deve applicarsi senza errori con `git apply --check`."
    )

TRACE_RE = re.compile(r'  File "(.+?)", line (\d+), in .+')

def traceback_frames(stderr: str):
    frames = []
    for row in stderr.splitlines():
        m = TRACE_RE.match(row)
        if m:
            path, line = m.group(1), int(m.group(2))
            ctx = "(file not found)"
            p = Path(path)
            if p.exists():
                lines = p.read_text().splitlines()
                ctx = "\n".join(lines[max(0, line-3): line+2])
            frames.append({"file": path, "line": line, "context": ctx})
    return frames

KEEP_RE = re.compile(r"^(?:diff --git |index |new file mode |deleted file mode |similarity index |rename (?:from|to) |--- |\+\+\+ |@@ |[ \+\-].*|\\ No newline at end of file)")

def clean_patch(raw: str) -> str:
    txt = re.sub(r"```(?:diff|patch)?\s*|```", "", raw)
    lines = [l.lstrip() for l in txt.splitlines() if not l.strip() or KEEP_RE.match(l.lstrip())]
    return "\n".join(lines) + "\n" if any(l.startswith("diff --git ") for l in lines) else ""

# ─── LLM calls ─────────────────────────────────────────────────
def ask_llm(prompt_json: str, max_tok: int = 1024) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT()},
            {"role": "user",    "content": prompt_json},
        ],
        temperature=0.0,
        max_tokens=max_tok,
    )
    return resp.choices[0].message.content.strip()

# ─── Apply diff ─────────────────────────────────────────────────
def apply_diff(diff: str) -> bool:
    df = Path("patch.diff"); df.write_text(diff)
    if run(["git", "apply", "--check", str(df)])[0] == 0 and run(["git", "apply", str(df)])[0] == 0:
        return True
    if run(["git", "apply", "-3", str(df)])[0] == 0:
        print("Patch applicata con git apply -3")
        return True
    if run(["patch", "-p1", "--fuzz", "3", "-i", str(df)])[0] == 0:
        print("Patch applicata con patch --fuzz=3")
        return True
    return False

# ─── Main loop ─────────────────────────────────────────────────
def main():
    for i in range(1, MAX_ITERS + 1):
        log("iter_start", iter=i)
        print(f"===== Iterazione {i}/{MAX_ITERS} =====")

        # Train
        log("train_start", iter=i)
        rc_train, _, _ = train()
        log("train_end", iter=i, rc=rc_train)
        if rc_train != 0:
            sys.exit("Errore in training")

        # Evaluate
        log("eval_start", iter=i)
        rc_eval, out_eval, err_eval = evaluate()
        log("eval_end", iter=i, rc=rc_eval)
        if rc_eval == 0:
            log("success", iter=i)
            print("Tutti i test passano. ✅")
            sys.exit(0)

        # Request diff
        prompt = {
            "goal": f"F1-macro ≥ {cfg['metrics']['f1_macro']}",
            "error_trace": err_eval,
            "test_output": out_eval,
            "file_context": traceback_frames(err_eval),
            "current_evaluate_py": Path("src/evaluate.py").read_text(),
            "config": cfg,
        }
        diff = clean_patch(ask_llm(json.dumps(prompt, indent=2), max_tok=1024))
        if not diff or len(diff.encode()) > MAX_PATCH_BYTES or not diff.startswith("diff --git a/src/"):
            log("patch_reject", iter=i, reason="invalid_or_too_big")
            print("Patch rifiutata: invalida o troppo grande.")
            continue

        log("patch_attempt", iter=i, size=len(diff))
        print("Patch ricevuta, provo ad applicarla…")
        if apply_diff(diff):
            rc2, _, _ = evaluate()
            if rc2 == 0:
                subprocess.run(["git", "add", "src/evaluate.py"], stdout=subprocess.DEVNULL)
                subprocess.run(["git", "commit", "-m", f"auto-patch diff: iter {i}"], stdout=subprocess.DEVNULL)
                log("patch_accept", iter=i, mode="diff", size=len(diff))
                print("Patch valida, test superati. ✅ Committato.")
                sys.exit(0)
            else:
                log("patch_reject", iter=i, reason="tests_fail")
                print("Patch applicata ma test KO; revert e fallback full-file.")
                run(["git", "restore", "src/evaluate.py"])
        else:
            log("patch_reject", iter=i, reason="apply_failed")
            print("Patch non applicabile; fallback full-file.")

        # Full‑file fallback
        full_prompt = {
            "instruction": (
                "Riscrivi interamente il file src/evaluate.py come codice Python valido, "
                "rispettando PEP8 e mantenendo import e funzioni. "
                "Restituisci SOLO il contenuto eseguibile senza diff né markdown."
            ),
            "current_evaluate_py": Path("src/evaluate.py").read_text(),
            "config": cfg,
        }
        full_text = ask_llm(json.dumps(full_prompt, indent=2), max_tok=2048)
        full_text = re.sub(r"```(?:python)?\s*|```", "", full_text)
        Path("src/evaluate.py").write_text(full_text)
        log("patch_accept", iter=i, mode="full_file", chars=len(full_text))
        print("evaluate.py riscritto, rieseguo evaluate()…")

        rc3, _, err3 = evaluate()
        if rc3 == 0:
            subprocess.run(["git", "add", "src/evaluate.py"], stdout=subprocess.DEVNULL)
            subprocess.run(["git", "commit", "-m", f"auto-patch full-file: iter {i}"], stdout=subprocess.DEVNULL)
            log("patch_accept", iter=i, mode="full_file", chars=len(full_text))
            print("File completo valido, test superati. ✅")
            sys.exit(0)
        else:
            last = err3.splitlines()[-1] if err3 else "unknown"
            log("patch_reject", iter=i, reason="fullfile_tests_fail", msg=last)
            print("Full-file ancora KO; continuo. →", last)

    print("Massimo iterazioni raggiunto; necessita intervento umano ✋")
    sys.exit(4)

if __name__ == "__main__":
    main()

