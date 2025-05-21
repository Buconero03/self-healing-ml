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
    """Testo fisso che istruisce l'LLM a produrre un diff valido."""
    return (
        "Sei un assistente che corregge bug in progetti Python seguendo PEP8 e senza introdurre nuove dipendenze. "
        "Quando proponi una patch, restituisci **solo** il testo del diff in formato unified (git diff -U0) con percorsi relativi. "
        "Assicurati che i file esistano realmente nel repository e che il diff passi `git apply --check` senza errori."
    )

# === Utility di shell ========================================================

def run_command(cmd: list[str]):
    """Esegue un comando di shell e restituisce (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

# === Task di training / evaluation ==========================================

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

# === Analisi stacktrace ======================================================

def parse_traceback(stderr: str):
    """Estrae il frame di errore più profondo con contesto di codice."""
    pattern = re.compile(r'  File "(.+?)", line (\d+), in .+')
    for row in stderr.splitlines():
        m = pattern.match(row)
        if m:
            fname, lineno = m.group(1), int(m.group(2))
            snippet = "(file non trovato)"
            path = Path(fname)
            if path.exists():
                src = path.read_text().splitlines()
                start = max(0, lineno - 3)
                snippet = "\n".join(src[start:lineno + 2])
            return [{"file": fname, "line": lineno, "context": snippet}]
    return []

# === Prompt per l'LLM ========================================================

def list_repo_files(limit: int = 200):
    rc, out, _ = run_command(["git", "ls-files"])
    if rc == 0:
        files = out.splitlines()
        return files[:limit] + (["... (troncato) ..."] if len(files) > limit else [])
    return []

def prepare_prompt(stderr: str, stdout: str):
    prompt = {
        "goal": f"Assicurati che F1-macro ≥ {cfg['metrics']['f1_macro']} e che tutti i test pytest passino.",
        "error_trace": stderr,
        "test_output": stdout,
        "file_context": parse_traceback(stderr),
        "repo_files": list_repo_files(),
        "config": cfg,
    }
    return json.dumps(prompt, indent=2)

# === Pulizia dell'output LLM =================================================

def clean_patch(raw: str) -> str:
    """Rimuove testo extra mantenendo un diff git completo e valido."""
    # 1) elimina i blocchi ``` diff/patch ```
    txt = re.sub(r"```(?:diff|patch)?\s*", "", raw)
    txt = txt.replace("```", "")

    # 2) mantieni SOLO le righe che appartengono a un diff git valido
    keep = re.compile(
        r"^(?:"
        r"diff --git |index |new file mode |deleted file mode |similarity index |rename (?:from|to) |"  # header vari
        r"--- |\+\+\+ |@@ |[ +\-].*|\\ No newline at end of file"  # hunks e contenuto
        r")"
    )
    lines = [l.rstrip("\r") for l in txt.splitlines() if keep.match(l)]

    # 3) assicura newline finale (git apply richiede LF finale)
    return "\n".join(lines) + "\n"

# === Chiamata LLM ============================================================

def call_llm(prompt_json: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT()},
            {"role": "user",   "content": prompt_json},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

# === Applicazione patch ======================================================

def apply_patch(patch_text: str) -> bool:
    """Applica la patch con più tentativi: git apply, git apply -3 e patch --fuzz."""
    diff_file = Path("patch.diff")
    diff_file.write_text(patch_text)

    # ---- 1. git apply "as‑is" -------------------------------------------
    rc, out, err = run_command(["git", "apply", "--check", str(diff_file)])
    if rc == 0:
        rc2, out2, err2 = run_command(["git", "apply", str(diff_file)])
        if rc2 == 0:
            return True
        print("git apply failed:", err2.strip() or out2.strip())
    else:
        print("git apply --check failed:", err.strip() or out.strip())

    # ---- 2. git apply con merge a 3 vie (-3) -----------------------------
    rc3, out3, err3 = run_command(["git", "apply", "-3", str(diff_file)])
    if rc3 == 0:
        print("Patch applicata con git apply -3 (risoluzione fuzzy).")
        return True
    print("git apply -3 failed:", err3.strip() or out3.strip())

    # ---- 3. patch classico con fuzz --------------------------------------
    rc4, out4, err4 = run_command(["patch", "-p1", "--fuzz", "3", "-i", str(diff_file)])
    if rc4 == 0:
        print("Patch applicata con patch --fuzz=3.")
        return True
    print("patch --fuzz failed:", err4.strip() or out4.strip())

    # Nessun metodo ha funzionato
    return False

# === Loop principale =========================================================

def main():
    for i in range(1, MAX_ITERS + 1):
        print(f"===== Iterazione {i}/{MAX_ITERS} =====")

        # Training
        rc, _, err = train()
        if rc != 0:
            print("Errore in training:", err)
            sys.exit(1)

        # Evaluation
        rc, out, err = evaluate()
        if rc == 0:
            print("Tutti i test passano. ✅")
            sys.exit(0)

        print("Valutazione fallita, preparo prompt per l'LLM…")
        prompt_json = prepare_prompt(err, out)

        # LLM
        raw_patch = call_llm(prompt_json)
        patch = clean_patch(raw_patch)
        if not patch.strip():
            print("LLM non ha restituito un diff valido.")
            sys.exit(2)

        print("Patch ricevuta, provo ad applicarla…")
        if not apply_patch(patch):
            sys.exit(3)

        # Commit
        subprocess.run(["git", "checkout", "self-healing"], stdout=subprocess.DEVNULL)
        subprocess.run(["git", "add", "-A"], stdout=subprocess.DEVNULL)
        subprocess.run(["git", "commit", "-m", f"auto-patch: iterazione {i}"], stdout=subprocess.DEVNULL)
        print("Patch applicata e committata. ↩️\n")

    print("Raggiunto il numero massimo di iterazioni; necessario intervento umano.")
    sys.exit(4)

# ============================================================================
if __name__ == "__main__":
    main()

