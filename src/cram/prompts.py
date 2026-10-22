"""Generic prompts that probe a document from many angles: 24 requests x 20 response
instructions = 480 prompts, as in the paper. They never contain evaluation questions: the
cache is trained before any question is known."""

REQUESTS = [
    "Summarize the document's main subject and structure.",
    "What is this document about?",
    "List the key facts stated in the document.",
    "Quote the first sentence of the document.",
    "Quote the last sentence of the document.",
    "What names, numbers or dates appear in the document?",
    "Describe how the document is organized.",
    "What kind of document is this?",
    "Restate the most important detail in the document.",
    "Which entities does the document mention, and how are they related?",
    "Repeat a passage from the middle of the document.",
    "What questions could someone answer with this document?",
    "Describe the writing style of the document.",
    "List the items the document enumerates.",
    "What is the most specific piece of information in the document?",
    "Give an example of a fact from the document.",
    "Recite the second sentence of the document.",
    "What patterns repeat throughout the document?",
    "Which detail in the document would be easiest to forget?",
    "Continue the document from its beginning.",
    "What does the document say near its end?",
    "Name three things the document mentions.",
    "Extract a few records from the document exactly as written.",
    "Explain the document to someone who has not read it.",
]

INSTRUCTIONS = [
    "Answer briefly.",
    "Answer in one sentence.",
    "Answer in detail.",
    "Use the exact wording of the document where possible.",
    "Answer with a list.",
    "Be specific.",
    "Start with the most important point.",
    "Answer in a few words.",
    "Quote the document.",
    "Include any numbers.",
    "Answer in two sentences.",
    "Give as many details as you can.",
    "Be precise.",
    "Answer in plain language.",
    "Mention names exactly as written.",
    "Keep the original order.",
    "Answer step by step.",
    "Use complete sentences.",
    "Answer with examples from the text.",
    "Answer as a short paragraph.",
]

PROMPTS = [f"\n\n{r} {i}\nAnswer:" for r in REQUESTS for i in INSTRUCTIONS]  # 480 prompts
