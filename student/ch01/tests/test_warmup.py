import torch
from torch import Tensor
from student.ch01.warmup import (NeuralNetwork,
    ToyDataset,
    normalization_helper,
    compute_accuracy)
from torch.utils.data import DataLoader

def test_shape_normalization_helper():
    x: Tensor = torch.tensor([[1.0,2.0,3.0]])
    out: Tensor = normalization_helper(x)
    assert x.shape == out.shape


def test_std_case_normalization_helper():
    x:Tensor = torch.tensor([[1.0,2.0,3.0]])
    out: Tensor = normalization_helper(x)
    assert torch.allclose(out.mean(dim=-1), torch.tensor([0.0]), atol=1e-6)



def test_normalization_handles_zero_std_case():
    x = torch.tensor([[2.0, 2.0, 2.0]])

    out = normalization_helper(x)

    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 0.0]]))



def test_output_shape_neural_network():
    model = NeuralNetwork(50, 30)
    x1: Tensor = torch.rand((5, 50))
    out1: Tensor = model(x1)
    assert out1.shape == (5, 30)

    model = NeuralNetwork(50, 20)
    x2: Tensor = torch.rand((1,50))
    out2: Tensor = model(x2)
    assert out2.shape == (1, 20)

    model = NeuralNetwork(50, 3)
    x3: Tensor = torch.rand((1,50))
    out3: Tensor = model(x3)
    assert out3.shape == (1, 3)

def test_gradient_flow():
    model = NeuralNetwork(50, 3)
    x = torch.rand((1,50))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()

    assert all(param.grad is not None for param in model.parameters())


def test_output_dtype_neural_network():
    model = NeuralNetwork(50, 3)
    x = torch.rand((1,50))
    logits = model(x)
    assert logits.dtype == torch.float32


def test_deterministic_neural_network():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    x = torch.rand((1,50))

    logits1 = model(x)
    
    torch.manual_seed(123)
    model2 = NeuralNetwork(50, 3)
    x2 = torch.rand((1,50))
    logits2 = model2(x2)
    assert torch.allclose(logits1, logits2)

def test_batch_size_independence():
    model = NeuralNetwork(50, 3)
    x1: Tensor = torch.rand((3,50))
    out1: Tensor = model(x1)
    row_1 = x1[0:1]
    out2 = model(row_1)
    assert torch.allclose(out1[0], out2[0])



def test_compute_accuracy():
    model = NeuralNetwork(50, 3)
    x = torch.rand((3,50))
    y = torch.tensor([0, 1, 2])
    dataloader = DataLoader(ToyDataset(x, y), batch_size=2, shuffle=False)
    accuracy = compute_accuracy(model, dataloader)
    assert accuracy <= 1.0 and accuracy >= 0.0


def test_compute_accuracy_perfect_accuracy():
    model = NeuralNetwork(50, 3)
    x = torch.rand((3,50))
    out = model(x)
    predictions = torch.argmax(out, dim=1)
    dataset = ToyDataset(x, predictions)
    dataloader = DataLoader(dataset)
    accuracy = compute_accuracy(model, dataloader)
    assert accuracy == 1.0
