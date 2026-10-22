"""Two attention functions plugged into Hugging Face models.

student: attention over a compact document cache (n learned slots, each with a bias) plus
         the ordinary causal cache of the tokens that follow the document. The two branches
         are merged through their log-normalizers, which is exactly one softmax over both.
capture: standard SDPA attention that also hands each layer's (post-RoPE) queries to a
         callback; used to collect teacher queries and to score the original cache.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

STUDENT, CAPTURE = "cram", "cram-capture"
_active: dict = {}


def student_attention(module, query, key, value, attention_mask=None, scaling=None, **kwargs):
    keys, values, bias, valid = _active["cache"].slots(module.layer_idx)  # [Hkv, n, d], ..., [Hkv, n]
    b, hq, t, d = query.shape
    hkv = keys.shape[0]
    groups = hq // hkv
    # Query heads sharing a KV head are adjacent, so grouping them is a reshape: no repeated K/V copies.
    q = query.float().reshape(b, hkv, groups * t, d)  # scores and normalizers in fp32, as in the paper
    scale = scaling if scaling is not None else d ** -0.5

    # Document branch: learned slots and biases, padded slots masked out.
    doc = torch.einsum("bkxd,knd->bkxn", q, keys) * scale + bias[None, :, None, :]
    doc = doc.masked_fill(~valid[None, :, None, :], float("-inf"))
    doc_lse = torch.logsumexp(doc, -1)
    # A head left with no slots has an all -inf row; its weight below is 0, so zero its NaNs.
    doc_out = torch.einsum("bkxn,knd->bkxd", torch.softmax(doc, -1), values).nan_to_num(0.0)

    # New-token branch: ordinary causal attention over the question and generated tokens.
    s = key.shape[2]
    new = torch.einsum("bkxd,bksd->bkxs", q, key.float()) * scale
    causal = torch.ones(t, s, dtype=torch.bool, device=q.device).tril(s - t).repeat(groups, 1)
    new = new.masked_fill(~causal, float("-inf"))
    new_lse = torch.logsumexp(new, -1)
    new_out = torch.einsum("bkxs,bksd->bkxd", torch.softmax(new, -1), value.float())

    total = torch.logaddexp(doc_lse, new_lse)
    out = torch.exp(doc_lse - total)[..., None] * doc_out + torch.exp(new_lse - total)[..., None] * new_out
    return out.reshape(b, hq, t, d).to(query.dtype).transpose(1, 2).contiguous(), None


def capture_attention(module, query, key, value, attention_mask, **kwargs):
    _active["sink"](module, query, key, value)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register(STUDENT, student_attention)
AttentionInterface.register(CAPTURE, capture_attention)
AttentionMaskInterface.register(CAPTURE, sdpa_mask)  # the student builds its own causal mask


@contextlib.contextmanager
def student(model, cache):
    """Run `model` against a compact document cache."""
    with _mode(model, STUDENT, cache=cache):
        yield


@contextlib.contextmanager
def capture(model, sink: Callable):
    """Run `model` normally while `sink(module, query, key, value)` sees every layer's attention."""
    with _mode(model, CAPTURE, sink=sink):
        yield


@contextlib.contextmanager
def _mode(model, name: str, **state):
    previous = model.config._attn_implementation
    model.config._attn_implementation = name
    _active.update(state)
    try:
        yield
    finally:
        model.config._attn_implementation = previous
        _active.clear()
