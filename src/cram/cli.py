"""cram compact FILE [--ratio 16] [--model M]    study a text file into FILE.cram
cram ask FILE.cram "question"                    answer from the compact cache"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import CompactCache, Config, compact, generate

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cram", description="Cram a long document into a tiny KV cache.")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compact", help="study a text file into a compact cache")
    c.add_argument("document")
    c.add_argument("-o", "--out", help="output path (default: DOCUMENT.cram)")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--ratio", type=float, default=16.0, help="compression ratio in KV pairs (default 16)")
    c.add_argument("--steps", type=int, nargs=2, default=[1000, 1000], metavar=("STAGE1", "STAGE2"))
    a = sub.add_parser("ask", help="answer a question from a compact cache")
    a.add_argument("cache")
    a.add_argument("question")
    a.add_argument("--model", help="defaults to the model the cache was made with")
    a.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.command == "compact":
        model, tok = _load(args.model, device)
        text = Path(args.document).read_text(encoding="utf-8")
        cache = compact(model, tok, text, Config(ratio=args.ratio, stage1_steps=args.steps[0],
                                                 stage2_steps=args.steps[1]))
        out = args.out or f"{args.document}.cram"
        cache.save(out, model=args.model)
        print(f"saved {cache.entries:,} KV pairs to {out}")
    else:
        cache = CompactCache.load(args.cache, device=device)
        model, tok = _load(args.model or cache.meta.get("model", DEFAULT_MODEL), device)
        print(generate(model, tok, cache, f"\n\nQuestion: {args.question}\nAnswer:", args.max_new_tokens).strip())


def _load(name: str, device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, AutoTokenizer.from_pretrained(name)


if __name__ == "__main__":
    main()
