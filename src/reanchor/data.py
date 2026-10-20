"""Datasets that combine MNIST images with one-hot hint vectors."""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def _wrong_label(label: int, generator: torch.Generator | None) -> int:
    offset = torch.randint(1, 10, (), generator=generator).item()
    return int(label + offset) % 10


def _one_hot(label: int) -> torch.Tensor:
    hint = torch.zeros(10)
    hint[label] = 1
    return hint


class HintedMNIST(Dataset):
    """MNIST with a 10-dim hint appended to each flattened image.

    group='A': hint is the correct label.
    group='B': hint is all zeros.
    group='C': hint is a uniformly random wrong label.
    group='mix': each sample gets A with probability p_a, else C.
    """

    def __init__(self, root: str, train: bool, group: str, p_a: float = 0.5,
                 download: bool = False, seed: int | None = None):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        self.group = group
        self.p_a = p_a
        # 固定 seed 时，C/mix 的 hint 生成可复现（评估集用）；None 则每样本重新随机（训练集用）。
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)

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
            hint = _one_hot(_wrong_label(label, self.generator))
        elif self.group == "mix":
            if torch.rand((), generator=self.generator).item() < self.p_a:
                hint = _one_hot(label)
            else:
                hint = _one_hot(_wrong_label(label, self.generator))
        else:
            raise ValueError(f"Unknown group: {self.group}")

        return torch.cat([image_flat, hint]), label

