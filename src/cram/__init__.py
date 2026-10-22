"""cram: train a tiny KV cache for a long document at test time, then answer anything from it.

    cache = cram.compact(model, tokenizer, document, cram.Config(ratio=16))
    cram.generate(model, tokenizer, cache, "\\n\\nQuestion: ...?\\nAnswer:")
"""

from .core import CompactCache, Config, Study, compact, compression, evict, forward, generate, study
from .prompts import PROMPTS

__all__ = ["CompactCache", "Config", "PROMPTS", "Study", "compact", "compression", "evict", "forward", "generate",
           "study"]
__version__ = "0.1.0"
