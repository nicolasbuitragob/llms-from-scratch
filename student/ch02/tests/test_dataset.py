import pytest


pytestmark = pytest.mark.skip(reason="Chapter 2 dataset not implemented yet")


def test_input_target_shift_for_one_context_window():
    # Example:
    # token_ids = [5, 9, 12, 4]
    # context_length = 3
    # input_ids should be [5, 9, 12]
    # target_ids should be [9, 12, 4]
    raise NotImplementedError


def test_dataset_length_depends_on_context_length_and_stride():
    # Use a small fixed token list and compute the expected number of windows.
    raise NotImplementedError


def test_dataset_item_shapes_match_context_length():
    # One item should return tensors shaped (context_length,).
    raise NotImplementedError


def test_dataset_item_dtypes_are_integer_token_ids():
    # Token IDs should use an integer dtype suitable for embedding lookup.
    raise NotImplementedError


def test_dataloader_batch_shape():
    # Batched input and target tensors should have shape:
    # (batch_size, context_length)
    raise NotImplementedError


def test_short_text_behavior_is_explicit():
    # Decide whether too-short text returns zero examples or raises a clear error.
    raise NotImplementedError
