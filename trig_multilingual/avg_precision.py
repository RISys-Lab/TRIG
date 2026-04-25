#!/usr/bin/env python3
"""Compute per-language average of character_ned, token_ned, sentence_accuracy from TRIG result JSON."""

import argparse
import json
from pathlib import Path

METRICS = ("character_ned", "token_ned", "sentence_accuracy")


def avg_three(lang_block: dict) -> float:
    vals = [float(lang_block[m]) for m in METRICS]
    return sum(vals) / len(vals)


def process_json(json_path: Path) -> None:
    json_path = json_path.resolve()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_lang = data.get("by_language")
    if not isinstance(by_lang, dict):
        raise ValueError("JSON missing 'by_language' object")

    lines = []
    for code, block in by_lang.items():
        if not isinstance(block, dict):
            continue
        missing = [m for m in METRICS if m not in block]
        if missing:
            raise KeyError(f"{code}: missing keys {missing}")
        lines.append((code, avg_three(block)))

    out_path = json_path.with_suffix(".txt")
    out_path.write_text(
        "\n".join(f"{code}\t{avg:.10f}" for code, avg in lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path} ({len(lines)} languages)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "json_path",
        type=Path,
        help="Path to results JSON (e.g. results_gemini_parallel10_anytext.json)",
    )
    args = p.parse_args()
    if not args.json_path.is_file():
        raise SystemExit(f"Not a file: {args.json_path}")
    process_json(args.json_path)


if __name__ == "__main__":
    main()
