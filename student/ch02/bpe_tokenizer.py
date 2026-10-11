"""Chapter 2, section 2.5: use the GPT-2 tokenizer from tiktoken.

Implement the methods in order: __init__, encode, decode.
Use tiktoken's existing tokenizer. You do not need a custom vocabulary.
Then explore whitespace, unfamiliar words, and special tokens in the tests.
"""

from collections.abc import Collection
from collections.abc import Set as AbstractSet
from typing import Literal

import tiktoken


class TiktokenTokenizer:
    """Exercise structure for encoding and decoding with GPT-2 BPE."""

    def __init__(self):
        # TODO: Load the GPT-2 encoding with tiktoken and store it on self.
        self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")

    def encode(
        self, text: str, *, allowed_special: AbstractSet[str] | Literal["all"] = set(), disallowed_special:Collection[str]   | Literal["all"] = 'all'
    ) -> list[int]:
        """Encode text. Allow special tokens only when explicitly requested."""
        return self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs with the stored encoding."""
        return self.tokenizer.decode(token_ids)


if __name__ == "__main__":
    tokenizer = TiktokenTokenizer()
    tokens = tokenizer.encode("Hello, world! dfsjhbvhsalduhjfb", allowed_special="all")   
    print(tokens)
    text = tokenizer.decode(tokens)
    print(text)