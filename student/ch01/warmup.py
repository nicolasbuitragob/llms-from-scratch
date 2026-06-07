import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits




class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features: Tensor = X
        self.labels: Tensor = y
    def __getitem__(self, index):
        one_x: Tensor = self.features[index]
        one_y: Tensor = self.labels[index]
        return one_x, one_y
    def __len__(self):
        return self.labels.shape[0]


def normalization_helper(tensor):
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, keepdim=True)
    eps = 1e-5
    std = std + eps
    return (tensor - mean) / std 


def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for features, labels in dataloader:
        features = normalization_helper(features)
        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
    return correct / total



if __name__ == "__main__":
    model = NeuralNetwork(50, 3)    


    xtrain: Tensor = torch.rand((1, 50))
    xtest: Tensor = torch.rand((1, 50))
    ytrain: Tensor = torch.randint(0, 3, (1,))
    ytest: Tensor = torch.randint(0, 3, (1,))


    train_ds = ToyDataset(xtrain, ytrain)
    test_ds = ToyDataset(xtest, ytest)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
