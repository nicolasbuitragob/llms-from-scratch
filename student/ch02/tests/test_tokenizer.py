import pytest


pytestmark = pytest.mark.skip(reason="Chapter 2 tokenizer not implemented yet")


def test_tokenize_separates_words_and_punctuation():
    # Example expectation:
    # "Hello, world!" -> ["Hello", ",", "world", "!"]
    raise NotImplementedError


def test_vocab_contains_expected_tokens_and_special_tokens():
    # Check that normal tokens and special tokens receive stable integer IDs.
    raise NotImplementedError


def test_encode_returns_token_ids():
    # encode("Hello") should return a list of integers.
    raise NotImplementedError


def test_decode_reconstructs_simple_text():
    # For a simple sentence, decode(encode(text)) should match the behavior
    # you intentionally designed.
    raise NotImplementedError


def test_unknown_token_behavior_is_explicit():
    # Decide whether unknown tokens map to <|unk|> or raise a clear error.
    raise NotImplementedError
