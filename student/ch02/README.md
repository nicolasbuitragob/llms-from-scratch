# Chapter 2: Working With Text Data

## Goal

Turn raw text into the input embeddings that a GPT model can process.

```text
raw text
   ↓ tokenizer
token IDs
   ↓ sliding windows
input and target batches: (batch, context)
   ↓ token embedding lookup
token embeddings: (batch, context, embedding dimension)
   +
position embeddings: (context, embedding dimension)
   ↓
GPT input embeddings: (batch, context, embedding dimension)
```

You implement the exercises. The official repository is reference material.
Ask for a hint before you ask for a complete solution.

## Files

```text
student/ch02/
  README.md
  the-verdict.txt
  tokenizer.py
  dataset.py
  embeddings.py                  # create during Exercise 4
  tests/
    test_tokenizer.py
    test_dataset.py
    test_embeddings.py           # create during Exercise 4
```

Do not change the files under `LLMs-from-scratch/ch02/`.

## Source Material

Read and use these files in this order:

1. `LLMs-from-scratch/ch02/01_main-chapter-code/ch02.ipynb`
2. `LLMs-from-scratch/ch02/01_main-chapter-code/dataloader.ipynb`
3. Optional after the main exercises:
   `LLMs-from-scratch/ch02/03_bonus_embedding-vs-matmul/`
4. Optional after you use `tiktoken`:
   `LLMs-from-scratch/ch02/05_bpe-from-scratch/`

Do not use `exercise-solutions.ipynb` until your implementation and tests pass.

## Working Method

For each exercise:

1. Read only the related book section.
2. Explain the input, output, and invariant in your own words.
3. Write or enable one small pytest test.
4. Predict the expected value, shape, and dtype.
5. Implement the smallest behavior that satisfies the test.
6. Run the focused test.
7. Run all Chapter 2 tests.
8. Review edge cases before you continue.

## Gate 1: Explain the Data Flow

Answer these questions before you implement code:

1. What is the difference between a token, a token ID, and an embedding?
2. Why do we need a vocabulary map and a reverse vocabulary map?
3. What should a simple tokenizer do with an unknown word?
4. Why can BPE process words that were not present as complete words in its
   vocabulary?
5. Why are training targets shifted one token forward?
6. For `[10, 11, 12, 13]` with context length `3`, what are the input and
   target?
7. What does `stride` change? What does it not change?
8. What shape should one dataset item have?
9. What shape should one DataLoader batch have?
10. Why does an input shaped `(B, T)` produce token embeddings shaped
    `(B, T, D)`?
11. Why can position embeddings shaped `(T, D)` be added to token embeddings
    shaped `(B, T, D)`?
12. What information is missing if we do not use positional embeddings?

## Exercise 1: Simple Tokenization and Vocabulary

Book sections: 2.1 through 2.4.

Work in `tokenizer.py` and `tests/test_tokenizer.py`.

Build these behaviors in order:

1. Split text into words and punctuation.
2. Build stable token-to-ID and ID-to-token maps.
3. Add `<|unk|>` and `<|endoftext|>`.
4. Encode text into integer IDs.
5. Decode IDs into readable text.
6. Define explicit behavior for unknown tokens.

Required tests:

- Punctuation is separated from words.
- Repeated tokens do not create duplicate vocabulary entries.
- Vocabulary IDs are stable for a fixed input.
- Both special tokens are present.
- `encode` returns integer IDs.
- `decode(encode(text))` follows the designed spacing rules.
- An unknown word uses `<|unk|>` or raises the chosen clear error.
- An invalid token ID fails clearly.

Completion gate:

- You can explain why encode and decode need opposite vocabulary maps.
- Every tokenizer test passes.

## Exercise 2: Byte Pair Encoding With `tiktoken`

Book section: 2.5.

Use the GPT-2 tokenizer from `tiktoken`. Do not implement BPE from scratch in
this exercise.

Explore and test:

1. Encoding and decoding simple text.
2. A word that was not present as a complete vocabulary entry.
3. Punctuation and whitespace behavior.
4. The `<|endoftext|>` token.
5. Allowed and disallowed special-token behavior.
6. Deterministic results for the same input.

Questions to answer:

- How does BPE differ from the simple word tokenizer?
- Why does GPT-2 BPE not need your custom `<|unk|>` token?
- Why can whitespace change the token IDs?
- Why must special-token handling be explicit?

Completion gate:

- You can compare simple tokenization and BPE with one concrete example.
- The `tiktoken` tests pass without copying its internal implementation.

## Exercise 3: Sliding Windows, Dataset, and DataLoader

Book section: 2.6.

Work in `dataset.py` and `tests/test_dataset.py`.

Build these behaviors in order:

1. Create windows from a fixed list of token IDs.
2. Use `context_length` to control each window size.
3. Use `stride` to control the distance between window starts.
4. Shift every target one token forward.
5. Implement a PyTorch Dataset.
6. Create DataLoader batches.
7. Define clear behavior for text that is too short.

Example:

```text
token IDs:       [5, 9, 12, 4]
context length:  3

input:           [5, 9, 12]
target:          [9, 12, 4]
```

Required tests:

- One input-target pair has the exact expected values.
- Dataset length is correct for different strides.
- Each item has shape `(context_length,)`.
- Input and target tensors use `torch.long`.
- Batch shape is `(batch_size, context_length)`.
- Shuffling behavior is intentional.
- Short-text behavior is explicit.
- Invalid context length or stride fails clearly.

Completion gate:

- You can calculate the number of windows before running the code.
- You can explain why the target uses the next token at every position.
- Every dataset test passes.

## Exercise 4: Token and Positional Embeddings

Book sections: 2.7 and 2.8.

Create `embeddings.py` and `tests/test_embeddings.py`.

Build these behaviors in order:

1. Create `torch.nn.Embedding(vocab_size, embedding_dim)` for token IDs.
2. Look up one ID and compare it with the correct weight-matrix row.
3. Embed a batch shaped `(B, T)`.
4. Create absolute position IDs with `torch.arange(T)`.
5. Create a position embedding table for the supported context length.
6. Add token embeddings and position embeddings.
7. Return GPT input embeddings shaped `(B, T, D)`.

Required tests:

- One token lookup equals the selected embedding row.
- Token embedding shape is `(B, T, D)`.
- Position IDs are `[0, 1, ..., T - 1]`.
- Position embedding shape is `(T, D)`.
- Position embeddings broadcast across the batch dimension.
- The final addition preserves `(B, T, D)`.
- A sequence longer than the supported context fails clearly.
- Seeded initialization is repeatable.

Questions to answer:

- Is an embedding lookup equivalent to a matrix multiplication with a one-hot
  vector?
- Which dimension is the vocabulary dimension?
- Which dimension is the learned vector dimension?
- Why does the same token ID have the same token embedding at every position?
- How do position embeddings make two occurrences of the same token different?

Completion gate:

- You can trace `(B, T)` to `(B, T, D)`.
- You can explain PyTorch broadcasting during the embedding addition.
- Every embedding test passes.

## Final Chapter Test

Run the focused tests while you work:

```bash
pytest student/ch02/tests -k tokenizer -q
pytest student/ch02/tests -k dataset -q
pytest student/ch02/tests -k embeddings -q
```

Then run the full Chapter 2 suite:

```bash
pytest student/ch02/tests -q
```

Chapter 2 is complete when:

- You can explain the full text-to-embedding flow without notes.
- The simple tokenizer and `tiktoken` differences are clear.
- Exact values, shapes, and dtypes have focused tests.
- Edge-case behavior is intentional.
- All Chapter 2 tests pass.
- You can explain every tensor transformation in your implementation.
