import pytest

from student.ch02.tokenizer import SimpleTokenizerV1


def test_tokenize_separates_words_and_punctuation():
    # Example expectation:
    # "Hello, world!" -> ["Hello", ",", "world", "!"]
    tokenizer = SimpleTokenizerV1("Hello, world!")
    assert tokenizer.split_text("Hello, world!") == ["Hello", ",", "world", "!"]
    assert tokenizer.split_text("This is a test.") == ["This", "is", "a", "test", "."]
    assert tokenizer.split_text("Multiple   spaces!") == ["Multiple", "spaces", "!"]
    assert tokenizer.split_text("Punctuation: test, test!") == ["Punctuation", ":", "test", ",", "test", "!"]
    assert tokenizer.split_text("what about question?") == ["what", "about", "question", "?"]
    


def test_vocab_contains_expected_tokens_and_special_tokens():
    # Check that normal tokens and special tokens receive stable integer IDs.
    tokenizer_1 = SimpleTokenizerV1("Hello, world!")
    vocab_1: dict[str, int] = tokenizer_1.vocab
    tokenizer_2 = SimpleTokenizerV1("Hello, world!")
    vocab_2: dict[str, int] = tokenizer_2.vocab
    assert vocab_1["Hello"] == vocab_2["Hello"]
    assert vocab_1[","] == vocab_2[","]
    assert vocab_1["<|unk|>"] == vocab_2["<|unk|>"]
    assert vocab_1["!"] == vocab_2["!"]



def test_encode_returns_token_ids():
    # encode("Hello") should return a list of integers.
    tokenizer = SimpleTokenizerV1("Hello, world!")
    encoded: list[int] = tokenizer.encode("Hello, world!")
    assert len(encoded) == 5  # "Hello", ",", "world", "!", "<|endoftext|>"
    assert encoded == [4,1,5,0,2]  # Assuming the vocab assigns these IDs in order
    assert all(isinstance(token_id, int) for token_id in encoded)  # All elements should be integers
    assert isinstance(encoded, list)


def test_decode_reconstructs_simple_text():
    # For a simple sentence, decode(encode(text)) should match the behavior
    # you intentionally designed.
    tokenizer = SimpleTokenizerV1("Hello, world!")
    text = "Hello, world!"
    encoded: list[int] = tokenizer.encode(text)
    decoded: str = tokenizer.decode(encoded)
    assert decoded == text  # The decoded text should match the original text.


def test_unknown_token_behavior_is_explicit():
    # Decide whether unknown tokens map to <|unk|> or raise a clear error.
    tokenizer = SimpleTokenizerV1("Hello, world!")
    unk_tokens: list[str] = ["Unknown", "token", "test"]
    encoded: list[int] = tokenizer.encode(" ".join(unk_tokens))
    unk_token_id: int = tokenizer.vocab["<|unk|>"]

    assert tokenizer.vocab["<|unk|>"] in encoded  
    for token in unk_tokens:
        token_id:int = encoded[unk_tokens.index(token)]
        assert token_id == unk_token_id    

def test_id_invalid_should_fail():
    text = "This is a sample text for tokenization."
    tokenizer = SimpleTokenizerV1(text)
    invalid_id: int = 999
    with pytest.raises(ValueError):
        tokenizer.decode(tokens=[invalid_id])  # Decoding an invalid ID should raise an error.
 
