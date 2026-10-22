"""Latent Compaction (Chen et al., 2026): train a small document KV cache at test time.

    init     preservers: the globally best entries by KVZip+ score (3/4 of the budget);
             anchors: the first entry and the most-attended entries (1/4 of the rest);
             diversifiers: farthest-first keys, for coverage of key space
    stage 1  per layer, fit anchors and diversifiers so that attention over the compact
             cache matches the full cache's outputs and log-normalizers (preservers frozen)
    stage 2  end to end, distill the full-cache model's next-token distributions on generic
             prompts into the whole compact cache
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache

from . import attention, teacher
from .prompts import PROMPTS


class CompactCache(nn.Module):
    """Per layer and KV head: n slots of (key, value, bias). Heads may use different numbers of
    slots, so each layer is padded to its largest head and `valid` marks the real slots."""

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor], valid: list[torch.Tensor],
                 frozen: list[torch.Tensor], length: int):
        super().__init__()
        self.keys = nn.ParameterList([nn.Parameter(k.float()) for k in keys])
        self.values = nn.ParameterList([nn.Parameter(v.float()) for v in values])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(k.shape[:2], device=k.device)) for k in keys])
        for i, (v, f) in enumerate(zip(valid, frozen)):
            self.register_buffer(f"valid{i}", v)
            self.register_buffer(f"frozen{i}", f)  # preservers: not trained in stage 1
        self.length = length  # tokens in the original document; later tokens are positioned after it
        self.meta: dict = {}

    def slots(self, layer: int):
        return self.keys[layer], self.values[layer], self.bias[layer], getattr(self, f"valid{layer}")

    def frozen(self, layer: int) -> torch.Tensor:
        return getattr(self, f"frozen{layer}")

    @property
    def entries(self) -> int:
        """Logical KV pairs stored, over all layers and KV heads."""
        return sum(int(getattr(self, f"valid{i}").sum()) for i in range(len(self.keys)))

    def save(self, path, **meta) -> None:
        """Save the cache; `meta` (e.g. model="Qwen/Qwen3-0.6B") is stored alongside."""
        torch.save({"state": self.state_dict(), "length": self.length, "layers": len(self.keys), "meta": meta},
                   path)

    @classmethod
    def load(cls, path, device="cpu") -> CompactCache:
        blob = torch.load(path, map_location=device, weights_only=True)
        s, n = blob["state"], blob["layers"]
        cache = cls([s[f"keys.{i}"] for i in range(n)], [s[f"values.{i}"] for i in range(n)],
                    [s[f"valid{i}"] for i in range(n)], [s[f"frozen{i}"] for i in range(n)], blob["length"])
        cache.load_state_dict(s)
        cache.meta = blob.get("meta", {})
        return cache


@dataclass
class Config:
    ratio: float = 16.0  # KV pairs in the document cache / KV pairs kept
    preserver_share: float = 0.75  # of the budget, chosen globally by KVZip+
    anchor_share: float = 0.25  # of the remaining anchor + diversifier slots
    stage1_steps: int = 1000
    stage2_steps: int = 1000
    stage1_lr: float = 3e-3
    stage2_lr: float = 1e-4
    stage1_batch: int = 128  # query positions per update
    stage2_batch: int = 4  # rollouts per update
    max_new: int = 24  # teacher continuation length
    top_k: int = 20  # teacher tokens kept individually in the distillation target
    max_queries: int = 4096  # teacher queries kept per layer for stage 1
    holdout: float = 0.1  # rollouts kept aside to pick the best stage-2 checkpoint (0: fixed schedule)
    eval_every: int = 100
    seed: int = 0


@dataclass
class Study:
    """What the full-cache model provides about one document; it does not depend on the budget."""

    doc_ids: torch.Tensor
    full: DynamicCache
    scores: list[torch.Tensor]  # KVZip+ score of every cached entry, [Hkv, N] per layer
    rollouts: list[teacher.Rollout]
    queries: list[torch.Tensor]  # teacher queries, [Hq, P, d] per layer


def study(model, tokenizer, document: str, config: Config | None = None, prompts: list[str] | None = None,
          log=print) -> Study:
    cfg = config or Config()
    log = log or (lambda *_: None)
    model.eval().requires_grad_(False)
    doc_ids = tokenizer(document, add_special_tokens=False, return_tensors="pt").input_ids[0]
    t0 = time.time()
    full = teacher.prefill(model, doc_ids)
    scores = teacher.kvzip_scores(model, tokenizer, full, doc_ids)
    rollouts, queries = teacher.rollouts(model, tokenizer, full, prompts or PROMPTS, cfg.max_new, cfg.top_k,
                                         cfg.max_queries, log)
    log(f"teacher: {len(doc_ids):,} document tokens, {len(rollouts)} rollouts ({time.time() - t0:.0f}s)")
    return Study(doc_ids, full, scores, rollouts, queries)


def compact(model, tokenizer, document: str | None = None, config: Config | None = None,
            prompts: list[str] | None = None, log=print, studied: Study | None = None) -> CompactCache:
    """Compress `document` into a CompactCache for `model`, before any question is known."""
    cfg = config or Config()
    log = log or (lambda *_: None)
    torch.manual_seed(cfg.seed)
    s = studied or study(model, tokenizer, document, cfg, prompts, log)
    cache = initialize(s.full, s.scores, s.queries, cfg)
    log(f"init: {cache.entries:,} of {len(s.doc_ids) * sum(x.shape[0] for x in s.scores):,} KV pairs "
        f"({cfg.ratio:g}x smaller)")
    t0 = time.time()
    stage1(model, s.full, cache, s.queries, cfg, log)
    log(f"stage 1 done ({time.time() - t0:.0f}s)")
    t0 = time.time()
    stage2(model, cache, s.rollouts, cfg, log)
    log(f"stage 2 done ({time.time() - t0:.0f}s)")
    return cache.requires_grad_(False)


def evict(model, tokenizer, document: str | None = None, ratio: float = 16.0,
          studied: Study | None = None) -> CompactCache:
    """The KVZip+ baseline: keep the globally best original entries, untrained, no biases."""
    if studied is None:
        doc_ids = tokenizer(document, add_special_tokens=False, return_tensors="pt").input_ids[0]
        full = teacher.prefill(model, doc_ids)
        scores = teacher.kvzip_scores(model, tokenizer, full, doc_ids)
    else:
        full, scores = studied.full, studied.scores
    return initialize(full, scores, None, Config(ratio=ratio, preserver_share=1.0)).requires_grad_(False)


def initialize(full: DynamicCache, scores: list[torch.Tensor], queries: list[torch.Tensor] | None,
               cfg: Config) -> CompactCache:
    n = full.get_seq_length()
    heads = scores[0].shape[0]
    budget = max(1, round(n / cfg.ratio))  # average slots per layer and KV head
    learned = 0 if cfg.preserver_share >= 1 else max(1, round(budget * (1 - cfg.preserver_share)))
    # Preservers: one global ranking of KVZip+ scores over all layers, heads and positions.
    flat = torch.cat([s.flatten() for s in scores])
    chosen = torch.zeros_like(flat, dtype=torch.bool)
    chosen[flat.topk((budget - learned) * len(scores) * heads).indices] = True
    chosen = chosen.view(len(scores), heads, n)

    keys, values, valid, frozen = [], [], [], []
    for layer in range(len(scores)):
        k, v = teacher.doc_kv(full, layer)
        picks = []
        for h in range(heads):
            pres = chosen[layer, h].nonzero().squeeze(1)
            extra = pres[:0]
            if learned:
                extra = _anchors_and_diversifiers(k[h], queries[layer] if queries else None, h, heads, learned,
                                                  cfg.anchor_share, pres)
            picks.append((pres, extra))
        width = max(1, max(len(p) + len(e) for p, e in picks))
        lk = torch.zeros(heads, width, k.shape[-1], device=k.device)
        lv = torch.zeros_like(lk)
        lvalid = torch.zeros(heads, width, dtype=torch.bool, device=k.device)
        lfrozen = torch.zeros_like(lvalid)
        for h, (pres, extra) in enumerate(picks):
            idx = torch.cat([pres, extra])
            lk[h, :len(idx)], lv[h, :len(idx)] = k[h, idx].float(), v[h, idx].float()
            lvalid[h, :len(idx)] = True
            lfrozen[h, :len(pres)] = True
        keys.append(lk)
        values.append(lv)
        valid.append(lvalid)
        frozen.append(lfrozen)
    return CompactCache(keys, values, valid, frozen, n)


def _anchors_and_diversifiers(k: torch.Tensor, q: torch.Tensor | None, head: int, heads: int, m: int,
                              anchor_share: float, taken: torch.Tensor) -> torch.Tensor:
    """m slots for one KV head: the first entry and the most-attended entries, then farthest-first keys."""
    n = len(k)
    free = torch.ones(n, dtype=torch.bool, device=k.device)
    free[taken] = False
    anchors = [0] if free[0] else []
    free[0] = False
    n_anchor = max(1, round(m * anchor_share))
    if q is not None and n_anchor > len(anchors):
        groups = q.shape[0] // heads
        qh = q[head * groups:(head + 1) * groups].to(k.device).float().flatten(0, 1)  # this KV head's queries
        attn = torch.zeros(n, device=k.device)
        for s in range(0, len(qh), 512):
            attn = torch.maximum(attn, torch.softmax(qh[s:s + 512] @ k.float().T * k.shape[-1] ** -0.5, -1).amax(0))
        attn[~free] = -1
        top = attn.topk(min(n_anchor - len(anchors), int(free.sum()))).indices
        anchors += top.tolist()
        free[top] = False
    picked = torch.tensor(anchors, dtype=torch.long, device=k.device)
    # Farthest-first traversal from the anchors (preservers are not part of the reference set).
    kf = k.float()
    dist = torch.cdist(kf, kf[picked]).amin(1) if len(picked) else torch.full((n,), float("inf"), device=k.device)
    extra = []
    for _ in range(min(m - len(picked), int(free.sum()))):
        i = int(dist.masked_fill(~free, -1).argmax())
        extra.append(i)
        free[i] = False
        dist = torch.minimum(dist, (kf - kf[i]).norm(dim=1))
    return torch.cat([picked, torch.tensor(extra, dtype=torch.long, device=k.device)])


def stage1(model, full: DynamicCache, cache: CompactCache, queries: list[torch.Tensor], cfg: Config, log) -> None:
    """Fit each layer's attention outputs and log-normalizers to the full cache's (preservers frozen)."""
    for layer in range(len(cache.keys)):
        k_full, v_full = (t.float() for t in teacher.doc_kv(full, layer))
        q = queries[layer].to(k_full.device).float()  # [Hq, P, d]
        groups = q.shape[0] // k_full.shape[0]
        scale = q.shape[-1] ** -0.5
        with torch.no_grad():  # targets over the whole document cache
            out_t, lse_t = [], []
            for s in range(0, q.shape[1], 256):
                logits = q[:, s:s + 256] @ k_full.repeat_interleave(groups, 0).transpose(-1, -2) * scale
                out_t.append(torch.softmax(logits, -1) @ v_full.repeat_interleave(groups, 0))
                lse_t.append(torch.logsumexp(logits, -1))
            out_t, lse_t = torch.cat(out_t, 1), torch.cat(lse_t, 1)
            o_mu, o_sd = out_t.mean((1, 2), keepdim=True), out_t.std((1, 2), keepdim=True).clamp_min(1e-4)
            z_mu, z_sd = lse_t.mean(1, keepdim=True), lse_t.std(1, keepdim=True).clamp_min(1e-4)
        params = [cache.keys[layer], cache.values[layer], cache.bias[layer]]
        trainable = ~cache.frozen(layer)
        opt = torch.optim.AdamW(params, lr=cfg.stage1_lr, weight_decay=0.0)
        for step in range(cfg.stage1_steps):
            idx = torch.randint(0, q.shape[1], (cfg.stage1_batch,), device=q.device)
            out, lse = _doc_attention(q[:, idx], *cache.slots(layer), groups, scale)
            loss = F.mse_loss((out - o_mu) / o_sd, (out_t[:, idx] - o_mu) / o_sd) + \
                F.mse_loss((lse - z_mu) / z_sd, (lse_t[:, idx] - z_mu) / z_sd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in params:  # preservers stay exactly as they were
                p.grad[~trainable] = 0
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if layer % 7 == 0 or layer == len(cache.keys) - 1:
            log(f"  stage 1 layer {layer + 1}/{len(cache.keys)}: normalized attention loss {loss.item():.4f}")


def _doc_attention(q, keys, values, bias, valid, groups, scale):
    """Attention of queries [Hq, B, d] over the compact slots alone: outputs and log-normalizers."""
    hq, n, d = q.shape
    q = q.reshape(hq // groups, groups * n, d)  # query heads grouped with their KV head
    logits = q @ keys.transpose(-1, -2) * scale + bias[:, None]
    logits = logits.masked_fill(~valid[:, None], float("-inf"))
    out = torch.softmax(logits, -1) @ values
    return out.reshape(hq, n, d), torch.logsumexp(logits, -1).reshape(hq, n)


def stage2(model, cache: CompactCache, rollouts: list[teacher.Rollout], cfg: Config, log) -> None:
    """Distill the teacher's next-token distributions on the rollouts into the whole compact cache.

    A held-out share of the rollouts (generic prompts only) picks the checkpoint to keep, starting
    with the stage-1 cache itself, so this stage cannot make the held-out fit worse."""
    order = torch.randperm(len(rollouts), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    n_val = int(len(rollouts) * cfg.holdout)
    val = [rollouts[i] for i in order[:n_val]]
    train = [rollouts[i] for i in order[n_val:]]
    best, best_step, best_state = (_held_out(model, cache, val), 0, _snapshot(cache)) if val else (None, 0, None)
    opt = torch.optim.AdamW(cache.parameters(), lr=cfg.stage2_lr, weight_decay=0.0)
    cache.requires_grad_(True)
    running = None
    for step in range(1, cfg.stage2_steps + 1):
        batch = [train[j] for j in torch.randint(0, len(train), (cfg.stage2_batch,)).tolist()]
        loss = distill_loss(model, cache, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        opt.step()
        running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()
        if val and (step % cfg.eval_every == 0 or step == cfg.stage2_steps):
            held = _held_out(model, cache, val)
            if held < best:
                best, best_step, best_state = held, step, _snapshot(cache)
        if step % 200 == 0 or step == cfg.stage2_steps:
            log(f"  stage 2 step {step}/{cfg.stage2_steps}: KL + LM {running:.4f}"
                + (f", held out {best:.4f} (best at step {best_step})" if val else ""))
    if val:
        cache.load_state_dict(best_state)


@torch.no_grad()
def _held_out(model, cache: CompactCache, rollouts: list[teacher.Rollout], batch: int = 4) -> float:
    losses = [distill_loss(model, cache, rollouts[i:i + batch]).item() * len(rollouts[i:i + batch])
              for i in range(0, len(rollouts), batch)]
    return sum(losses) / len(rollouts)


def _snapshot(cache: CompactCache) -> dict:
    return {k: v.detach().clone() for k, v in cache.state_dict().items()}


def distill_loss(model, cache: CompactCache, rollouts: list[teacher.Rollout]) -> torch.Tensor:
    """Grouped KL over the teacher's top-k tokens plus the tail mass, plus cross-entropy on its greedy
    tokens, averaged over rollouts. Rollouts are right-padded into one batch: attention is causal,
    so no real token ever sees a pad."""
    seqs = [r.ids[:-1] for r in rollouts]  # teacher forcing; the last token is only a target
    ids = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    ids = ids.to(model.device)
    positions = torch.arange(cache.length, cache.length + ids.shape[1], device=model.device)[None].expand(len(ids), -1)
    with attention.student(model, cache):
        hidden = _decoder(model)(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state
    # Only the continuation positions are scored, so only they go through the (large) vocabulary head.
    spans = [hidden[i, r.prompt_len - 1:len(seqs[i])] for i, r in enumerate(rollouts)]
    logp = torch.log_softmax(model.get_output_embeddings()(torch.cat(spans)).float(), -1)
    losses = []
    for r, lp in zip(rollouts, logp.split([len(s) for s in spans])):
        top_ids, p_logp, p_tail = (t.to(lp.device) for t in (r.top_ids, r.top_logp, r.tail_logp))
        r_top = lp.gather(1, top_ids)
        r_tail = torch.log1p(-r_top.exp().sum(1).clamp(max=1 - 1e-6))
        kl = (p_logp.exp() * (p_logp - r_top)).sum(1) + p_tail.exp() * (p_tail - r_tail)
        losses.append((kl - r_top[:, 0]).mean())  # the teacher's greedy token is its top-1
    return torch.stack(losses).mean()


def _decoder(model):
    """The transformer under a causal-LM head (e.g. Qwen3ForCausalLM.model)."""
    return getattr(model, "model", None) or model.get_decoder()


def forward(model, cache: CompactCache, ids: torch.Tensor, past: DynamicCache | None = None) -> torch.Tensor:
    """Logits for `ids` ([T] or [B, T]) read after the compacted document; positions continue after it."""
    batched = ids.dim() == 2
    ids = ids if batched else ids[None]
    past = past if past is not None else DynamicCache(config=model.config)
    start = cache.length + past.get_seq_length()
    positions = torch.arange(start, start + ids.shape[1], device=model.device)[None].expand(len(ids), -1)
    with attention.student(model, cache):
        logits = model(input_ids=ids, position_ids=positions, past_key_values=past, use_cache=True).logits
    return logits if batched else logits[0]


@torch.no_grad()
def generate(model, tokenizer, cache: CompactCache | None, prompt: str, max_new_tokens: int = 32,
             full: DynamicCache | None = None) -> str:
    """Greedy answer to `prompt` read against a compact cache (or, for reference, the full cache)."""
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device)
    past = DynamicCache(config=model.config) if cache is not None else full
    n0 = past.get_seq_length()
    out = []
    for _ in range(max_new_tokens):
        if cache is not None:
            logits = forward(model, cache, ids, past)[-1]
        else:
            logits = model(input_ids=ids[None], past_key_values=past, use_cache=True).logits[0, -1]
        token = int(logits.argmax())
        if token == tokenizer.eos_token_id:
            break
        out.append(token)
        ids = torch.tensor([token], device=model.device)
    if cache is None:
        full.crop(n0)
    return tokenizer.decode(out, skip_special_tokens=True)


def compression(document_tokens: int, cache: CompactCache) -> float:
    layers = len(cache.keys)
    heads = cache.keys[0].shape[0]
    return document_tokens * layers * heads / cache.entries if cache.entries else math.inf
