## literary_cli.py

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from ai_core import LiteraryAI


def read_any(path: Optional[str]) -> str:
    if path in (None, "-"):
        return sys.stdin.read()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".txt", ".md"}:
        return p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(str(p))
        return "\n".join(para.text for para in doc.paragraphs)
    if p.suffix.lower() == ".pdf":
        from pdfminer.high_level import extract_text
        return extract_text(str(p))
    # Fallback: try utf-8
    return p.read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="LiteraryAI CLI: extract characters, SVO actions, emotion curve from English literature"
    )
    ap.add_argument("input", nargs="?", default="-", help="Path to input file (.txt/.md/.docx/.pdf) or '-' for stdin")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for transformers")
    ap.add_argument("--sm", action="store_true", help="Use small spaCy model (fallback if trf not installed)")
    ap.add_argument("--maxlen", type=int, default=None, help="Trim input to first N characters (speed debug)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ai = LiteraryAI(prefer_trf=not args.sm, gpu=(not args.cpu), max_doc_len=args.maxlen)
    text = read_any(args.input)

    summary = ai.analyze(text, args.out)

    print("\n=== Done. Artifacts: ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("summary.json: ", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
