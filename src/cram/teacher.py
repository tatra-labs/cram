"""What the full-cache teacher provides: the document's KV cache, a KVZip+ importance score for
every cached entry, and generic rollouts (queries for stage 1, next-token targets for stage 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from . import attention

RECONSTRUCT = "\n\nRepeat the previous context exactly.\n"


@dataclass
class Rollout:
    ids: torch.Tensor  # prompt followed by the teacher's greedy continuation
    prompt_len: int
    top_ids: torch.Tensor  # [T, k] teacher's top-k tokens at each continuation step
    top_logp: torch.Tensor  # [T, k] their log-probabilities
    tail_logp: torch.Tensor  # [T] log of the probability mass outside the top k


def layers(model):
    return model.model.layers


@torch.no_grad()
def prefill(model, doc_ids: torch.Tensor) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    model(input_ids=doc_ids[None].to(model.device), past_key_values=cache, use_cache=True)
    return cache


def doc_kv(cache: DynamicCache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keys (with RoPE) and values of the document in one layer: [Hkv, N, d] each."""
    return cache.layers[layer].keys[0], cache.layers[layer].values[0]


@torch.no_grad()
def kvzip_scores(model, tokenizer, cache: DynamicCache, doc_ids: torch.Tensor, chunk: int = 256) -> list[torch.Tensor]:
    """KVZip+: while the model repeats the document, the most attention each cached entry gets,
    scaled by its projected value norm over the query's hidden-state norm. [Hkv, N] per layer."""
    n = cache.get_seq_length()
    repeat = tokenizer(RECONSTRUCT, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = torch.cat([repeat, doc_ids.cpu()])[None].to(model.device)
    norms, scores = {}, [None] * len(layers(model))

    def remember_norm(layer_idx):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            norms[layer_idx] = hidden[0].float().norm(dim=-1)  # [T]
        return hook

    def sink(module, query, key, value):
        q, k = query[0].float(), key[0].float()  # [Hq, T, d], [Hkv, N + T, d]
        hq, t, d = q.shape
        groups = hq // k.shape[0]
        w_o = module.o_proj.weight.float().view(-1, hq, d)  # [E, Hq, d]
        v_doc = value[0, :, :n].float()  # [Hkv, N, d]
        # ||W_o^h v_i|| for every query head h and document entry i: [Hq, N]
        vnorm = torch.stack([(v_doc[h // groups] @ w_o[:, h].T).norm(dim=-1) for h in range(hq)])
        best = torch.zeros(hq, n, device=q.device)
        keys = k.repeat_interleave(groups, 0)
        for s in range(0, t, chunk):
            qc = q[:, s:s + chunk]
            logits = qc @ keys.transpose(-1, -2) * module.scaling  # [Hq, c, N + T]
            rows = torch.arange(s, s + qc.shape[1], device=q.device)[:, None]
            future = torch.arange(k.shape[1] - n, device=q.device)[None] > rows
            logits[..., n:] = logits[..., n:].masked_fill(future, float("-inf"))
            weights = torch.softmax(logits, -1)[..., :n] / norms[module.layer_idx][s:s + chunk, None]
            best = torch.maximum(best, weights.amax(1))
        scores[module.layer_idx] = (best * vnorm).view(-1, groups, n).amax(1)  # max over grouped heads

    hooks = [layer.register_forward_pre_hook(remember_norm(i), with_kwargs=True) for i, layer in enumerate(layers(model))]
    try:
        with attention.capture(model, sink):
            model(input_ids=ids, past_key_values=cache, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
        cache.crop(n)
    return scores


@torch.no_grad()
def rollouts(model, tokenizer, cache: DynamicCache, prompts: list[str], max_new: int = 24, top_k: int = 20,
             max_queries: int = 4096, log=None) -> tuple[list[Rollout], list[torch.Tensor]]:
    """Greedy teacher continuations of generic prompts from independent forks of the document
    cache. Returns the rollouts and, per layer, the teacher queries seen along the way [Hq, P, d]."""
    n = cache.get_seq_length()
    queries: list[list[torch.Tensor]] = [[] for _ in layers(model)]
    eos = tokenizer.eos_token_id

    def sink(module, query, key, value):
        queries[module.layer_idx].append(query[0].to(torch.bfloat16).cpu())

    out = []
    with attention.capture(model, sink):
        for j, prompt in enumerate(prompts):
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0].to(model.device)
            logits = model(input_ids=ids[None], past_key_values=cache, use_cache=True).logits[0, -1]
            cont, tops, logps, tails = [], [], [], []
            for _ in range(max_new):
                logp = torch.log_softmax(logits.float(), -1)
                top = logp.topk(top_k)
                tops.append(top.indices.cpu())
                logps.append(top.values.cpu())
                tails.append(torch.log1p(-top.values.exp().sum().clamp(max=1 - 1e-6)).cpu())
                token = int(top.indices[0])
                cont.append(token)
                if token == eos or len(cont) == max_new:
                    break
                logits = model(input_ids=torch.tensor([[token]], device=model.device), past_key_values=cache,
                               use_cache=True).logits[0, -1]
            cache.crop(n)
            out.append(Rollout(torch.cat([ids.cpu(), torch.tensor(cont)]), len(ids), torch.stack(tops),
                               torch.stack(logps), torch.stack(tails)))
            if log and (j + 1) % 40 == 0:
                log(f"  teacher rollouts {j + 1}/{len(prompts)}")
    pools = []
    for qs in queries:
        q = torch.cat(qs, dim=1)  # [Hq, P, d]
        if q.shape[1] > max_queries:  # keep evenly spaced positions
            q = q[:, torch.linspace(0, q.shape[1] - 1, max_queries).long()]
        pools.append(q)
    return out, pools
