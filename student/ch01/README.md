What is a tensor?

Your answer:
- A tensor is a datatype designed to storage vectors of multiple dimensions. It is the basic datatype unit for ML training

Correction:
- A tensor is a multi-dimensional array of numbers. In PyTorch, a tensor also has a shape, dtype, and device, and it can participate in automatic differentiation when gradient tracking is enabled.

What is a model / module?

Your answer:
- It is the blueprint/ computational graph that ddetermines the model architecture

Correction:
- A PyTorch module is an object, usually an `nn.Module`, that contains parameters and defines a `forward` computation. A model architecture is built by combining modules and tensor operations.

What does loss.backward() do?

Your answer:
- computes the gradients of models and updates params

Correction:
- `loss.backward()` computes gradients and stores them in each trainable parameter's `.grad` field. It does not update parameters. The optimizer updates parameters when `optimizer.step()` is called.

Training-step sequence:
- `loss.backward()` fills gradients.
- `optimizer.step()` updates parameters using those gradients.
- `optimizer.zero_grad()` clears old gradients before the next step.

What is next-token prediction?

Your answer:
- the process of ingesting a corpus of text converting it to numbers and based on that output the most likely token next in the corpus of ingest data

Correction:
- Next-token prediction means that, given previous tokens, the model predicts a probability distribution over the vocabulary for the next token. During training, the prediction is compared against the actual next token from the text.
