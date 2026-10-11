"""
Tests for the GPTDatasetV1 class.
"""

import pytest

from student.ch02.dataset import GPTDatasetV1


def test_input_target_shift_for_one_context_window():
    dataset = GPTDatasetV1(token_ids=[5, 9, 12, 4], context_length=3)
    input_target_pairs = dataset.create_input_target_pairs(token_ids=[5, 9, 12, 4], context_length=3)
    input_ids = input_target_pairs["input_ids"]
    target_ids = input_target_pairs["target_ids"]
    assert input_ids == [[5, 9, 12]]
    assert target_ids == [[9, 12, 4]]


@pytest.mark.parametrize("tokens,context_length,stride,windows,target_windows", [
    ([1, 2, 3, 4, 5, 6], 3, 1, [[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[2, 3, 4], [3, 4, 5], [4, 5, 6]]),
    ([1,2,3,4,5,6,7,8,9,10], 3, 2, [[1,2,3], [3,4,5], [5,6,7], [7,8,9]], [[2,3,4], [4,5,6], [6,7,8], [8,9,10]]),
])
def test_dataset_length_depends_on_context_length_and_stride(tokens, context_length, stride, windows, target_windows):
    # Use a small fixed token list and compute the expected number of windows.
   
    dataset = GPTDatasetV1(token_ids=tokens, context_length=context_length)
    input_target_pairs = dataset.create_input_target_pairs(token_ids=tokens, 
    context_length=context_length, 
    stride=stride)    

    input_ids = input_target_pairs["input_ids"]
    target_ids = input_target_pairs["target_ids"]

    assert input_ids == windows
    assert target_ids == target_windows

@pytest.mark.skip(reason="Dataset stride exercise not implemented yet")
def test_dataset_item_shapes_match_context_length():
    # One item should return tensors shaped (context_length,).
    token_ids = [10, 20, 30, 40, 50]
    context_length = 4
    dataset = GPTDatasetV1(token_ids=token_ids, context_length=context_length)
    input_target_pairs = dataset.create_input_target_pairs(token_ids=token_ids, context_length=context_length)
    input_ids = input_target_pairs["input_ids"]
    target_ids = input_target_pairs["target_ids"]
    assert len(input_ids) == context_length
    assert len(target_ids) == context_length

@pytest.mark.skip(reason="Dataset stride exercise not implemented yet")
def test_dataset_item_dtypes_are_integer_token_ids():
    # Token IDs should use an integer dtype suitable for embedding lookup.
    token_ids = [100, 200, 300, 400]
    context_length = 3
    dataset = GPTDatasetV1(token_ids=token_ids, context_length=context_length)
    input_target_pairs = dataset.create_input_target_pairs(token_ids=token_ids, context_length=context_length)
    input_ids = input_target_pairs["input_ids"]
    target_ids = input_target_pairs["target_ids"]
    assert all(isinstance(token_id, int) for token_id in input_ids)
    assert all(isinstance(token_id, int) for token_id in target_ids)

@pytest.mark.skip(reason="Dataset stride exercise not implemented yet")
def test_dataloader_batch_shape():
    # Batched input and target tensors should have shape:
    # (batch_size, context_length)
    raise NotImplementedError

@pytest.mark.skip(reason="Dataset stride exercise not implemented yet")
def test_short_text_behavior_is_explicit():
    # Decide whether too-short text returns zero examples or raises a clear error.
    raise NotImplementedError
