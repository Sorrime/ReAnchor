"""Feasibility check for ReAnchor v0.1.

Trains a two-layer MLP on a 50/50 mix of groups A (correct hint) and C
(wrong hint), then reports accuracy and mean confidence on groups A, B,
and C of the test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reanchor.data import HintedMNIST
from reanchor.model import HintedMLP


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    confidence_sum = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            probabilities = torch.softmax(model(inputs), dim=1)
            confidence, predicted = probabilities.max(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            confidence_sum += confidence.sum().item()

    return correct / total, confidence_sum / total


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_set = HintedMNIST("data", train=True, group="mix", p_a=0.5)
    test_sets = {
        "A": HintedMNIST("data", train=False, group="A"),
        "B": HintedMNIST("data", train=False, group="B"),
        "C": HintedMNIST("data", train=False, group="C"),
    }

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loaders = {
        name: DataLoader(dataset, batch_size=512)
        for name, dataset in test_sets.items()
    }

    model = HintedMLP(hidden_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)

        results = {
            name: evaluate(model, loader, device)
            for name, loader in test_loaders.items()
        }

        parts = [f"epoch {epoch + 1:2d}/{num_epochs}  loss {average_loss:.4f}"]
        for name, (accuracy, confidence) in results.items():
            parts.append(f"{name} acc {accuracy:.4f} conf {confidence:.4f}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
