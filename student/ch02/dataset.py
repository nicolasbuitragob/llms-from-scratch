"""Chapter 2 GPT-style dataset exercises.

Implement this file after the tokenizer basics work.

Suggested public API:
- create_input_target_pairs(token_ids, context_length, stride)
- GPTDatasetV1

Each dataset item should return:

```text
input_ids:  tensor shaped (context_length,)
target_ids: tensor shaped (context_length,)
```

The target should be the input shifted one token forward.
"""


class GPTDatasetV1:
    """Placeholder for your Chapter 2 dataset implementation."""

    pass
