# Tutor Notes

Notes for the tutor. Read at the start of each session. Update at the end.

## Learner preferences

- Communication: ASD-STE100 Simplified Technical English.
- Prefers diagrams and visualizations over long text. Tufte style, clean typography.
  A simple question does not need an HTML artifact.
- Hints first. No code unless the learner asks for it.
- Goal is storage strength, not fluency: use retrieval practice and desirable difficulty.
- Pytest: learn basic unit and end-to-end testing, not pytest expertise.
- Code must be well designed: KISS, YAGNI, DRY. Imports at top of file.

## Session protocol

1. Recall: 2–3 questions on earlier topics (see review queue). No notes. Then give feedback.
2. Continue the current exercise from `study-plan.html`.
3. End: update `study-plan.html` statuses and this file.

## Progress log

| Date       | Topic                                | Result |
|------------|--------------------------------------|--------|
| (earlier)  | App A: tensors, modules, gradients   | Done. `student/ch01/warmup.py` + tests. |
| (earlier)  | Ch 1: pipeline notes, toy next-token | Done. Several corrections needed (see below). |
| (earlier)  | Ch 2.1–2.4: simple tokenizer         | Done. All tests pass. |
| (earlier)  | Ch 2.5: tiktoken BPE                 | Done. All tests pass. |
| 2026-10-10 | Ch 2.6: sliding windows / Dataset    | In progress. Loop body is empty. |
| 2026-10-11 | Ch 2.6: sliding windows              | Loop correct. Learned: expected values come from hand calculation, not a formula in the test; parametrize. Found the off-by-one only after writing targets by hand. Forgot the 3-window result once (recall slip). |

## Observed misconceptions (corrected in Ch 1)

- Thought `loss.backward()` also updates parameters. (It only fills `.grad`; `optimizer.step()` updates.)
- Thought tokens must be unique in the text. (Vocabulary entries are unique; tokens repeat.)
- Described a module as "the blueprint / computational graph". (A module holds parameters and defines `forward`.)

Check these again in recall questions. Storage strength is not yet proven.

## Recall results

| Date       | Question                          | Result |
|------------|-----------------------------------|--------|
| 2026-10-11 | Token / token ID / embedding      | Partial. Token and ID correct. Embedding vague: did not say "a row of a trainable weight matrix". |
| 2026-10-11 | What `loss.backward()` changes    | **Wrong again.** Said it changes params. Same misconception as Ch 1. Ask again next session. |
| 2026-10-11 | Why BPE needs no `<|unk|>`        | Partial. Said "byte pair encoding" but not the fallback to sub-words and single bytes. |
| 2026-10-11 | Order of zero_grad / backward / step | Wrong. Thinks backward computes the loss. Did not know `zero_grad()` or that gradients accumulate. |

## Open points for Ch 2.6 (dataset) review

Discuss with the learner. Do not fix for them.

- `test_dataset_length_depends_on_context_length_and_stride`: expected windows and expected
  values do not agree with each other (formula, `[[3, 4, 5]]` target for stride 1).
- `test_dataset_item_shapes_match_context_length` checks the number of windows, not the length of one item.
- `test_dataset_item_dtypes_are_integer_token_ids` checks a list of lists for `int`.
- `test_dataloader_batch_shape` raises `NotImplementedError` without a skip marker.
- Design: `GPTDatasetV1` is not yet a `torch.utils.data.Dataset` (`__len__`, `__getitem__`),
  and `create_input_target_pairs` takes the same args as `__init__` (duplicated state).
- `stride == 0` passes validation; `range` then raises an unclear "arg 3 must not be zero" error.
- `context_length == len(token_ids)` passes validation but gives 0 windows (each window needs `context_length + 1` tokens).

## Review queue (spaced retrieval)

- Why are targets shifted one token forward?
- Why does BPE not need `<|unk|>`?
- What does `loss.backward()` change? What does it not change?
- Token vs token ID vs embedding.
