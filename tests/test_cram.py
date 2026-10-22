from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import cram
from cram import teacher
from cram.core import CompactCache, Config, forward, initialize


class Chars:
    """A character-level stand-in for a real tokenizer."""

    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[2 + ord(c) % 60 for c in text]]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(48 + (i - 2) % 60) for i in ids)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, intermediate_size=128)
    return Qwen3ForCausalLM(cfg).eval()


DOC = "The phone number of ALICE is 0123456789. The phone number of BOB is 9876543210. " * 2


def test_full_slot_cache_reproduces_the_full_cache(model):
    tok = Chars()
    doc = tok(DOC).input_ids[0]
    full = teacher.prefill(model, doc)
    keys = [full.layers[i].keys[0] for i in range(2)]
    values = [full.layers[i].values[0] for i in range(2)]
    every = [torch.ones(k.shape[:2], dtype=torch.bool) for k in keys]
    cache = CompactCache(keys, values, every, every, len(doc))
    ids = tok("\nQuestion: BOB?").input_ids[0]
    with torch.no_grad():
        student = forward(model, cache, ids)
        reference = model(input_ids=ids[None], past_key_values=full, use_cache=True).logits[0]
    assert torch.allclose(student, reference, atol=1e-4)


def test_kvzip_scores_and_rollouts(model):
    tok = Chars()
    doc = tok(DOC).input_ids[0]
    full = teacher.prefill(model, doc)
    scores = teacher.kvzip_scores(model, tok, full, doc)
    assert [s.shape for s in scores] == [(2, len(doc))] * 2 and all(torch.isfinite(s).all() for s in scores)
    assert full.get_seq_length() == len(doc)  # the probe leaves the cache as it found it
    rollouts, queries = teacher.rollouts(model, tok, full, cram.PROMPTS[:3], max_new=5, top_k=4)
    assert len(rollouts) == 3 and rollouts[0].top_ids.shape[1] == 4
    assert queries[0].shape[0] == 4 and queries[0].shape[1] == sum(len(r.ids) - 1 for r in rollouts)
    assert full.get_seq_length() == len(doc)


def test_budget_and_frozen_preservers(model):
    tok = Chars()
    doc = tok(DOC).input_ids[0]
    full = teacher.prefill(model, doc)
    scores = teacher.kvzip_scores(model, tok, full, doc)
    _, queries = teacher.rollouts(model, tok, full, cram.PROMPTS[:2], max_new=3)
    cache = initialize(full, scores, queries, Config(ratio=4))
    assert cache.entries == round(len(doc) / 4) * 2 * 2  # budget x layers x KV heads
    learned = round(round(len(doc) / 4) * 0.25)
    assert int((~cache.frozen(0) & cache.slots(0)[3]).sum()) == learned * 2


def test_batched_distillation_matches_one_rollout_at_a_time(model):
    from cram.core import distill_loss

    tok = Chars()
    doc = tok(DOC).input_ids[0]
    full = teacher.prefill(model, doc)
    scores = teacher.kvzip_scores(model, tok, full, doc)
    rollouts, queries = teacher.rollouts(model, tok, full, cram.PROMPTS[:3], max_new=6)
    cache = initialize(full, scores, queries, Config(ratio=4))
    with torch.no_grad():
        together = distill_loss(model, cache, rollouts)
        apart = torch.stack([distill_loss(model, cache, [r]) for r in rollouts]).mean()
    assert len({len(r.ids) for r in rollouts}) > 1  # different lengths, so padding is exercised
    assert torch.allclose(together, apart, atol=1e-4)


def test_stage2_keeps_the_stage1_cache_when_held_out_loss_only_rises(model, monkeypatch):
    import cram.core as cc

    tok = Chars()
    doc = tok(DOC).input_ids[0]
    full = teacher.prefill(model, doc)
    scores = teacher.kvzip_scores(model, tok, full, doc)
    rollouts, queries = teacher.rollouts(model, tok, full, cram.PROMPTS[:10], max_new=4)
    cache = initialize(full, scores, queries, Config(ratio=4))
    before = cc._snapshot(cache)
    rising = iter([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(cc, "_held_out", lambda *args, **kwargs: next(rising))
    cc.stage2(model, cache, rollouts, Config(ratio=4, stage2_steps=3, holdout=0.5, eval_every=1), lambda *_: None)
    assert all(torch.equal(v, before[k]) for k, v in cache.state_dict().items())


def test_compact_end_to_end_and_save(model, tmp_path):
    tok = Chars()
    cfg = Config(ratio=4, stage1_steps=20, stage2_steps=4, max_new=4)
    cache = cram.compact(model, tok, DOC, cfg, prompts=cram.PROMPTS[:4], log=None)
    assert all(not p.requires_grad for p in cache.parameters())
    answer = cram.generate(model, tok, cache, "\nQuestion: BOB?", max_new_tokens=5)
    assert isinstance(answer, str)
    cache.save(tmp_path / "doc.cram")
    again = CompactCache.load(tmp_path / "doc.cram")
    assert again.entries == cache.entries and torch.equal(again.keys[0], cache.keys[0])
    baseline = cram.evict(model, tok, DOC, ratio=4)
    assert baseline.entries == round(len(tok(DOC).input_ids[0]) / 4) * 2 * 2
