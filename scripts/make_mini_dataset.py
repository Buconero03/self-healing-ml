#!/usr/bin/env python3
"""
Estrae n esempi random dal dataset completo e li salva in JSONL
per train/val più rapidi.
Uso:
    poetry run python scripts/make_mini_dataset.py \
        --in data/train.jsonl --out data/mini/train.jsonl --size 400
"""
import argparse, random, json, os, pathlib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",   dest="src",  required=True, help="file JSONL sorgente")
    p.add_argument("--out",  dest="dst",  required=True, help="file JSONL mini")
    p.add_argument("--size", dest="k",    type=int, default=500, help="n° record")
    args = p.parse_args()

    lines = pathlib.Path(args.src).read_text().splitlines()
    sample = random.sample(lines, min(args.k, len(lines)))
    os.makedirs(pathlib.Path(args.dst).parent, exist_ok=True)
    pathlib.Path(args.dst).write_text("\n".join(sample))
    print(f"Saved {len(sample)} rows ➜ {args.dst}")

if __name__ == "__main__":
    random.seed(42)
    main()

