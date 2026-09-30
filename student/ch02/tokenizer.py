"""Chapter 2 tokenizer exercises.

Implement this file yourself as you work through Chapter 2.

Suggested public API:
- tokenize(text)
- build_vocab(tokens, special_tokens=None)
- SimpleTokenizerV1

Keep the first version simple. The goal is to understand tokens, token IDs,
vocabularies, unknown tokens, and encode/decode behavior before moving to BPE.
"""
import re


class SimpleTokenizerV1:
    """Placeholder for your Chapter 2 tokenizer implementation."""
    def __init__(self, text:str):
        self.text:str = text
        self.regex_pattern: str = r"([,.]|\s+)"  # Regex pattern to split on punctuation and whitespace
        self.unk_token: str = '<|unk|>'
        self.end_of_text_token: str = '<|endoftext|>'
        self.vocab: dict[str,int] = self.build_vocab()

    def split_text(self, text:str) -> list[str]:
        """Split the input text into tokens based on the regex pattern."""
        text_split: list[str] = re.split(self.regex_pattern, text)
        word_list: list[str] = [word for word in text_split if word.strip()]  # Remove empty strings    

        return word_list

    def build_vocab(self) -> dict[str,int]:
        word_list:list[str] = self.split_text(self.text)
        word_list.append(self.unk_token)  # Add unknown token to the word list
        word_list.append(self.end_of_text_token)  # Add end-of-text token to the
        word_list:list[str] = list(set(word_list))
        word_list.sort() 
        vocab: dict[str, int] = {word: idx for idx, word in enumerate(word_list)} 
        return vocab
        
    def encode(self, text:str) ->list[int]:
        
        vocab_unk_id: int = self.vocab[self.unk_token]
        vocab_eot_id: int = self.vocab[self.end_of_text_token]

        text_list:list[str] = self.split_text(text)

        encoded:list[int] = [
            self.vocab.get(word, vocab_unk_id) for word in text_list
        ]
        encoded.append(vocab_eot_id)  # Append end-of-text token ID

        return encoded

    def decode(self, tokens:list[int]) -> str:
        id_to_word: dict[int, str] = {idx: word for word, idx in self.vocab.items()}
        decoded:list[str] = []
        for token in tokens:
            if token in id_to_word:
                decoded.append(id_to_word[token])
        decoded: str = ' '.join(decoded)  # Join the tokens back into a single string 
        #remove end-of-text token from the decoded string
        decoded = decoded.replace(self.end_of_text_token, '').strip()
        #remove spaces before punctuation
        decoded = re.sub(r'\s+([,.])', r'\1', decoded)

        return decoded
    
            

if __name__ == '__main__':
    # Example usage
    text = "This is a sample text for tokenization. awesome, people, and more."
    tokenizer = SimpleTokenizerV1(text)
    print("Vocabulary:", tokenizer.vocab)

    text_to_encode = "This is, a sample text for tokenization."
    
    print("Encoded:", tokenizer.encode(text_to_encode))
    print("Decoded:", tokenizer.decode(tokenizer.encode(text_to_encode)))
    