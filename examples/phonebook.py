"""Phonebook Lookup: after a document's cache is compressed, how many numbers can the model
still look up? Compares the full cache, KVZip+ eviction, and cram after each stage.

    python examples/phonebook.py --model Qwen/Qwen3-0.6B --tokens 4096 --ratios 8 16 32 --docs 2
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import cram
from cram import phonebook
from cram.core import Config, initialize, stage1, stage2


def accuracy(model, tok, cache, pairs, full=None):
    hits = sum(number in cram.generate(model, tok, cache, phonebook.question(name), 16, full=full).replace(" ", "")
               for name, number in pairs)
    return 100 * hits / len(pairs)


def plot(results, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    styles = {"KVZip+ (eviction)": ("#8A8F99", "o", "KVZip+ (eviction)"),
              "cram, stage 1": ("#1F5FBF", "D", "cram, stage 1"),
              "cram": ("#7FA7E0", "s", "cram, stage 1 + 2")}
    full = [r["accuracy"] for r in results if r["method"] == "full cache"]
    ax.axhline(sum(full) / len(full), color="#222", ls="--", lw=1.2, label="full cache")
    for method, (color, marker, label) in styles.items():
        rows = [r for r in results if r["method"] == method]
        ratios = sorted({r["ratio"] for r in rows})
        acc = [sum(r["accuracy"] for r in rows if r["ratio"] == x) / sum(r["ratio"] == x for r in rows) for x in ratios]
        ax.plot(ratios, acc, marker=marker, color=color, lw=2.2 if method != "KVZip+ (eviction)" else 1.6, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r["ratio"] for r in results if r["ratio"] > 1}))
    ax.set_xticklabels([f"{x:g}x" for x in ax.get_xticks()])
    ax.set_xlabel("KV cache compression")
    ax.set_ylabel("lookup accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(title, fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--ratios", type=float, nargs="+", default=[8, 16, 32])
    p.add_argument("--docs", type=int, default=2)
    p.add_argument("--questions", type=int, default=20)
    p.add_argument("--steps", type=int, nargs=2, default=[1000, 1000], metavar=("STAGE1", "STAGE2"))
    p.add_argument("--prompts", type=int, default=len(cram.PROMPTS))
    p.add_argument("--out", default="results/phonebook")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    results, out = [], Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def record(doc, method, ratio, acc, seconds=0.0):
        results.append({"doc": doc, "method": method, "ratio": ratio, "accuracy": acc, "seconds": round(seconds)})
        print(f"doc {doc}  {method:<18} {ratio:>4g}x  {acc:5.1f}%  ({seconds:.0f}s)", flush=True)
        (out.parent / f"{out.name}.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))

    for d in range(args.docs):
        doc, entries = phonebook.phonebook(tok, args.tokens, seed=d)
        pairs = phonebook.sample(entries, args.questions, seed=d)
        t0 = time.time()
        studied = cram.study(model, tok, doc, Config(), cram.PROMPTS[:args.prompts])
        study_seconds = time.time() - t0
        torch.cuda.empty_cache()  # stay well inside VRAM: on Windows an overfull GPU pages to system RAM
        record(d, "full cache", 1, accuracy(model, tok, None, pairs, full=studied.full))
        for ratio in args.ratios:
            evicted = cram.evict(model, tok, ratio=ratio, studied=studied)
            record(d, "KVZip+ (eviction)", ratio, accuracy(model, tok, evicted, pairs))
            cfg = Config(ratio=ratio, stage1_steps=args.steps[0], stage2_steps=args.steps[1])
            torch.manual_seed(cfg.seed)
            cache = initialize(studied.full, studied.scores, studied.queries, cfg)
            t0 = time.time()
            stage1(model, studied.full, cache, studied.queries, cfg, print)
            t1 = time.time()
            record(d, "cram, stage 1", ratio, accuracy(model, tok, cache, pairs), study_seconds + t1 - t0)
            stage2(model, cache, studied.rollouts, cfg, print)
            cache.requires_grad_(False)
            record(d, "cram", ratio, accuracy(model, tok, cache, pairs), study_seconds + time.time() - t0)
            del cache, evicted
            torch.cuda.empty_cache()
        del studied
        torch.cuda.empty_cache()

    name = args.model.split("/")[-1]
    plot(results, out.parent / f"{out.name}.png", f"Phonebook Lookup, {name}, {args.tokens // 1024}K-token documents")


if __name__ == "__main__":
    main()
