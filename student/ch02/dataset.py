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

    def __init__(self, token_ids: list[int], context_length: int, stride: int = 1):
        self.token_ids: list[int] = token_ids
        self.context_length: int = context_length


    def create_input_target_pairs(self,
        token_ids:list[int],
        context_length:int,
        stride:int = 1)-> dict[str, list[list[int]]]:
        
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer.")
        
        if context_length > len(token_ids):
            raise ValueError("context_length cannot be greater than the length of token_ids.")
       
        if stride < 0:
            raise ValueError("stride must be a non-negative integer.")

        input_ids:list[list[int]] = []
        target_ids:list[list[int]] = []

        for start_index in range(0, len(token_ids) - context_length, stride):
            input_window = token_ids[start_index:start_index + context_length]
            target_window = token_ids[start_index + 1:start_index + context_length + 1]
            input_ids.append(input_window)
            target_ids.append(target_window)


  


        return {"input_ids": input_ids, "target_ids": target_ids}


if __name__ == "__main__":
    # Example usage
    token_ids = [1,2,3,4,5,6,7,8,9,10]
    context_length = 5
    dataset = GPTDatasetV1(token_ids, context_length)
    result = dataset.create_input_target_pairs(token_ids, context_length)
    
    print("Input IDs:", result['input_ids'])
    print("Target IDs:", result['target_ids'])