# Chapter 1 Pipeline Notes

## Raw text
raw text refers to the entire corpus of text data for training

Correction:
Raw text is ordinary text before tokenization. During training it comes from
a corpus; during inference it can be just the user prompt.

## Tokens
tokens are the individual pieces of text that belong to words that are unique for training data
e.g understanding -> tokens: under, standing 

Correction:
Tokens are pieces of text produced by a tokenizer. They do not have to be
unique in the text; the same token can appear many times.

## Token IDs
token ids refer to the ids assingned to each token
{TokenId:token}
{1:under},{2:standing}

Correction:
Token IDs are the integer IDs assigned to tokens. The usual vocabulary map is
token -> ID, for example {"under": 1, "standing": 2}. A reverse map can also
exist for decoding IDs back into tokens.

## Embeddings
Are the numerical representation of a token/word/text
are learnt during training
it is a vector of n dimensions

Correction:
An embedding turns each token ID into a learned vector so the model can work
with dense numerical representations instead of raw integer IDs.

## Transformer
Is the main NN architecture used in LLM
A transformer can have a encoder module and decoder module
What makes the transformer arch so special is the self attention mechanism

Correction:
A transformer is the main neural-network architecture used in many LLMs. Some
transformers use encoders and decoders, but GPT-style LLMs use decoder-only
transformer blocks with causal self-attention.

## Logits
Logits are the weights during trraining

Correction:
Logits are not weights. Logits are the model's raw output scores before
softmax. The usual flow is: logits -> softmax -> probabilities.

## Probabilities
the probability distribution of each token when predicting next token

Correction:
The model outputs a probability distribution over the vocabulary. For
next-token generation, we usually use the probability distribution at the
final position in the input sequence.

## Next token
the output of the llm process

Correction:
The next token is selected from the probability distribution, either by taking
the highest-probability token or by sampling.

## Input and target shifting

For next-token training, the model learns from shifted token sequences.

Example:

```python
token_ids = [5, 9, 12]

input_ids = [5, 9]
target_ids = [9, 12]
```

Meaning:

```text
at position 0: input 5 should predict target 9
at position 1: input 9 should predict target 12
```

The rule is:

```text
input_ids: everything except the last token
target_ids: everything except the first token
```

My first intuition was to think about separate passes:

```text
[5] -> [9]
[5, 9] -> [12]
```

That is useful for understanding autoregressive generation, but during
training we usually shift the whole sequence by one token and train all
positions in parallel.

## My summary

Next token prediction. This is how it works:

You get a whole corpus of text, and you feed it into the LLM. The LLM is going to preprocess this corpus of text, get the token IDs to any preprocessing required. It's going to get an embeddings layer, the embeddings of this text, and it's going to output a probability distribution of all the tokens in the dictionary. It outputs a probability distribution that shows the most likely token given the corpus of text in the input.

Once you have that, you append this token at the end of the corpus (at the initial corpus), and then you feed it again into the same architecture, and then you get the next likely token. Now you have two likely tokens, and then you append them again to the same corpus. Then you do it again and again and again. That's how it works, the next token prediction. 

Correction:
During training, the model sees chunks or batches from the corpus and learns to
predict the next token at each position. During generation, the model receives a
prompt, predicts one next token, appends that token to the prompt, and repeats
the process.


Pre-training and fine-tuning are different because pre-training, the model learns by using a self-supervised method, meaning that it doesn't require any labels. The model just learns based on the previous inputs in the training data.

Fine-tuning does require labels. Fine-tuning goes after pre-training.

First The model learns to be a next token prediction. With fine-tuning, you train it to be an assistant.

Now you have labels teaching the model how to be an assistant or how to do any task or classification task. You do require labels for this approach.

Tokenizers are algorithms that split words into small subsets of characters. 
Embeddings are the numerical representation of characters, tokens, words, or corpus of texts.
Transformers are the LLM architecture that have decoder, encoder layer, and the self-attention mechanism.
