"""Datasets that combine MNIST images with one-hot hint vectors."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _wrong_label(label: int) -> int:
    offset = torch.randint(1, 10, ()).item()
    return (label + offset) % 10


def _one_hot(label: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(label), num_classes=10).float()


class HintedMNIST(Dataset):
    """MNIST with a 10-dim hint appended to each flattened image.

    group='A': hint is the correct label.
    group='B': hint is all zeros.
    group='C': hint is a uniformly random wrong label.
    group='mix': each sample gets A with probability p_a, else C.
    """

    def __init__(self, root: str, train: bool, group: str, p_a: float = 0.5, download: bool = True):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        self.group = group
        self.p_a = p_a

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int):
        image, label = self.mnist[index]
        image_flat = torch.flatten(image)

        if self.group == "A":
            hint = _one_hot(label)
        elif self.group == "B":
            hint = torch.zeros(10)
        elif self.group == "C":
            hint = _one_hot(_wrong_label(label))
        elif self.group == "mix":
            if torch.rand(()).item() < self.p_a:
                hint = _one_hot(label)
            else:
                hint = _one_hot(_wrong_label(label))
        else:
            raise ValueError(f"Unknown group: {self.group}")

        return torch.cat([image_flat, hint]), label

