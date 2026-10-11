"""BPE exercises. Remove one skip marker when you start that test.

Write each test body yourself. Start with the round-trip test.
"""

import pytest

from student.ch02.bpe_tokenizer import TiktokenTokenizer


def test_encode_decode_restores_original_text():
    # TODO: Encode "Hello, world!" and check that the IDs are integers.
    # Decode the IDs and assert that the result equals the original text.
    tokenizer = TiktokenTokenizer()
    text = "Hello, world!"
    encoded: list[int] = tokenizer.encode(text, allowed_special="all", disallowed_special=())
    assert all(isinstance(token_id, int) for token_id in encoded)  # All elements should be integers
    assert isinstance(encoded, list)
    decoded: str = tokenizer.decode(encoded)
    assert decoded == text  # The decoded text should match the original text.



def test_unfamiliar_word_round_trip():
    tokenizer = TiktokenTokenizer()
    text = "flibbertigizmohjbsdefue238djnabkc"
    encoded: list[int] = tokenizer.encode(text, allowed_special="all")
    length_of_encoded = len(encoded)
    decoded: str = tokenizer.decode(encoded)
    assert decoded == text  # Decoded text should match the original
    assert length_of_encoded > 0  # Ensure that the encoded list is not empty



def test_whitespace_and_punctuation_are_preserved():
    # TODO: Use text with leading spaces, repeated spaces, and a newline.
    # Assert exact equality after encoding and decoding.
    tokenizer = TiktokenTokenizer()
    text = "   Hello,   world!\nThis is a test."
    encoded: list[int] = tokenizer.encode(text, allowed_special=set(), disallowed_special=())
    decoded: str = tokenizer.decode(encoded)
    assert decoded == text  # Decoded text should match the original
    



def test_endoftext_is_rejected_by_default():
    # TODO: Encode text containing <|endoftext|> without permission.
    # Use pytest.raises to check for ValueError.
    tokenizer = TiktokenTokenizer()
    text = "This is a test <|endoftext|>."
    with pytest.raises(ValueError):
        tokenizer.encode(text)  # Should raise ValueError because <|endoftext|> is not allowed by default


def test_endoftext_can_be_explicitly_allowed():

    # TODO: Allow <|endoftext|> through the allowed_special argument.
    # Check that the marker encodes as one token and decodes to itself.
    tokenizer = TiktokenTokenizer()
    text = "This is a test <|endoftext|>."
    encoded: list[int] = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert len(encoded) > 0  # Ensure that the encoded list is not empty
    decoded: str = tokenizer.decode(encoded)
    assert decoded == text  # Decoded text should match the original 
    special_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    assert len(special_token_id) == 1  # <|endoftext|

def test_encoding_is_repeatable():
    # TODO: Encode the same text twice and assert equal ID lists.
    tokenizer = TiktokenTokenizer()
    text = "Hello, world!"
    encoded1: list[int] = tokenizer.encode(text, allowed_special="all")
    encoded2: list[int] = tokenizer.encode(text, allowed_special="all")
    assert encoded1 == encoded2  # Encoded lists should be equal
