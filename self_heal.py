#!/usr/bin/env python3
import os
import subprocess
import json
import sys
import re
from pathlib import Path
import yaml
from openai import OpenAI

# === Configurazione ==========================================================
CONFIG_PATH = Path("config/quality.yml")
if not CONFIG_PATH.exists():
    print(f"File di configurazione non trovato: {CONFIG_PATH}")
    sys.exit(1)

cfg = yaml.safe_load(CONFIG_PATH.read_text())
MAX_ITERS = cfg.get("self_heal", {}).get("max_iterations", 5)
MODEL     = cfg.get("model_name", "model.joblib")

# === OpenAI client ===========================================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Variabile di ambiente OPENAI_API_KEY mancante.")
    sys.exit(1)
client = OpenAI(api_key=api_key)

# === Prompt di sistema =======================================================

def SYSTEM_PROMPT() -> str:
    return (
        "Sei un assistente che corregge bug in progetti Python seguendo PEP8 e senza introdurre nuove dipendenze. "
        "Quando proponi una patch, restituisci solo il testo del diff in formato unified (git diff -U0) con percorsi relativi. "
        "Il diff deve applicarsi con `git apply --check` senza errori. "
        "Per modificare `src/evaluate.py` usa come contesto il codice inviato in `current_evaluate_py`. "
        "Inserisci sempre le intestazioni `diff --git`/`index`/`---`/`+++` e lascia una riga vuota prima di ogni blocco @@."
    )

# === Funzione di utilità per eseguire comandi ================================

def run_command(cmd: list[str]):
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr

# === Train / Evaluate ========================================================

def train():
    return run_command([
        "poetry", "run", "python", "src/train.py",
        "--config", str(CONFIG_PATH),
        "--model-out", MODEL,
    ])

def evaluate():
    return run_command([
        "poetry", "run", "python", "src/evaluate.py",
        "--config", str(CONFIG_PATH),
        "--model", MODEL,
    ])

# === Traceback parser ========================================================

def parse_traceback(stderr: str):
    pat = re.compile(r'  File "(.+?)", line (\d+), in .+')
    for row in stderr.splitlines():
        m = pat.match(row)
        if m:
            fname, lineno = m.group(1), int(m.group(2))
            snippet = "(file non trovato)"
            p = Path(fname)
            if p.exists():
                src = p.read_text().splitlines()
                start = max(0, lineno - 3)
                snippet = "\n".join(src[start:lineno + 2])
            return [{"file": fname, "line": lineno, "context": snippet}]
    return []

# === Prompt builder ==========================================================

def list_repo_files(limit: int = 200):
    rc, out, _ = run_command(["git", "ls-files"])
    if rc == 0:
        files = out.splitlines()
        return files[:limit] + (["... (troncato) ..."] if len(files) > limit else [])
    return []

def prepare_prompt(stderr: str, stdout: str):
    eval_path = Path("src/evaluate.py")
    current_eval = eval_path.read_text() if eval_path.exists() else ""

    prompt = {
        "goal": f"Assicurati che F1-macro ≥ {cfg['metrics']['f1_macro']} e che tutti i test pytest passino.",
        "error_trace": stderr,
        "test_output": stdout,
        "file_context": parse_traceback(stderr),
        "repo_files": list_repo_files(),
        "current_evaluate_py": current_eval,
        "config": cfg,
    }
    return json.dumps(prompt, indent=2)

# === Pulizia diff dall'LLM ===================================================

def clean_patch(raw: str) -> str:
    txt = re.sub(r"```(?:diff|patch)?\s*", "", raw).replace("```", "")
    keep = re.compile(
        r"^(?:"  
        r"diff --git |index |new file mode |deleted file mode |similarity index |rename (?:from|to) |"  
        r"--- |\+\+\+ |@@ |[ +\-].*|\\ No newline at end of file)"
    )
    lines = []
    for line in txt.splitlines():
        s = line.lstrip()
        if s == "":
            lines.append("")
            continue
        if keep.match(s):
            lines.append(s)
    if not any(l.startswith("diff --git ") for l in lines):
        return ""
    return "\n".join(lines) + "\n"

# === Chiamata LLM ============================================================

def call_llm(prompt_json: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT()},
            {"role": "user", "content": prompt_json},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()

# === Applicazione patch ======================================================

def apply_patch(patch_text: str):
    diff_file = Path("patch.diff")
    diff_file.write_text(patch_text)

    # Tentativo 1: git apply --check / git apply
    rc, out, err = run_command(["git", "apply", "--check", str(diff_file)])
    if rc == 0 and run_command(["git", "apply", str(diff_file)])[0] == 0:
        return True
    print("git apply --check failed:", err.strip() or out.strip())

    # Tentativo 2: git apply -3
    if run_command(["git", "apply", "-3", str(diff_file)])[0] == 0:
        print("Patch applicata con git apply -3.")
        return True

    # Tentativo 3: patch con fuzz
    rc3, out3, err3 = run_command(["patch", "-p1", "--fuzz", "3", "-i", str(diff_file)])
    if rc3 == 0:
        print("Patch applicata con patch --fuzz=3.")
        return True
    print("patch --fuzz failed:", err3.strip() or out3.strip())

    # Fallback full-file mode
    return None

# === Loop principale =========================================================

def main():
    for i in range(1, MAX_ITERS + 1):
        print(f"===== Iterazione {i}/{MAX_ITERS} =====")

        if train()[0] != 0:
            print("Errore in training")
            sys.exit(1)

        rc, out, err = evaluate()
        if rc == 0:
            print("Tutti i test passano. ✅")
            sys.exit(0)

        print("Valutazione fallita, preparo prompt per l'LLM…")
        patch = clean_patch(call_llm(prepare_prompt(err, out)))
        if not patch:
            print("LLM non ha restituito un diff valido.")
            sys.exit(2)

        print("Patch ricevuta, provo ad applicarla…")
        res = apply_patch(patch)
        if res is False:
            sys.exit(3)
        elif res is None:
            print("Patch non applicabile: passo al fallback file intero.")
            eval_src = Path("src/evaluate.py").read_text()
            full_prompt = {
                "instruction": "Riscrivi interamente `src/evaluate.py`, correggendo gli errori di indentazione e rispettando PEP8. Restituisci solo il file completo senza diff.",
                "current_evaluate_py": eval_src,
                "config": cfg
            }
            raw_full = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT()},
                    {"role": "user", "content": json.dumps(full_prompt, indent=2)}
                ],
                temperature=0.0,
                max_tokens=2048
            ).choices[0].message.content.strip()
            Path("src/evaluate.py").write_text(raw_full)
            print("File `src/evaluate.py` riscritto in full-file mode.")

        subprocess.run(["git", "checkout", "self-healing"], stdout=subprocess.DEVNULL)
        subprocess.run(["git", "add", "-A"], stdout=subprocess.DEVNULL)
        subprocess.run(["git", "commit", "-m", f"auto-patch: iterazione {i}"], stdout=subprocess.DEVNULL)
        print("Patch applicata e committata. ↩️\n")

    print("Raggiunto il numero massimo di iterazioni; necessario intervento umano.")
    sys.exit(4)

if __name__ == "__main__":
    main()

