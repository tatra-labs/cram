"""Phonebook Lookup: a document of name -> number lines; ask for one number at a time."""

from __future__ import annotations

import random
import string


def phonebook(tokenizer, tokens: int, seed: int = 0) -> tuple[str, dict[str, str]]:
    """A document of about `tokens` tokens of 'The phone number of NAME is NUMBER.' lines,
    with unique 10-letter uppercase names and 10-digit numbers."""
    rng = random.Random(seed)
    entries: dict[str, str] = {}
    lines: list[str] = []
    while True:
        name = "".join(rng.choices(string.ascii_uppercase, k=10))
        if name in entries:
            continue
        entries[name] = "".join(rng.choices(string.digits, k=10))
        lines.append(f"The phone number of {name} is {entries[name]}.")
        if len(lines) % 20 == 0 and len(tokenizer("\n".join(lines), add_special_tokens=False).input_ids) >= tokens:
            return "\n".join(lines), entries


def question(name: str) -> str:
    return f"\n\nQuestion: What is the phone number of {name}?\nAnswer:"


def sample(entries: dict[str, str], k: int, seed: int = 0) -> list[tuple[str, str]]:
    return random.Random(seed + 1).sample(sorted(entries.items()), k)
