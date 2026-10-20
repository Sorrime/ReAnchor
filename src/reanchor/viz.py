"""Train on MNIST + hint, then look at the digits the model got wrong.

Design notes: the old train_one_group mixed "train the model", "score it" and
"print a line". This file keeps those apart so a plot can reuse the same
probabilities:

    train_model()   -> model + per-epoch history
    predict()       -> raw (confidence, predicted, labels) for any loader
    accuracy_of()   -> accuracy / mean confidence from those arrays
    plot_*()        -> display only, no math

matplotlib keeps its own window after plt.show() returns, so the figure stays
alive even though this script finishes.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from reanchor.data import HintedMNIST
from reanchor.model import HintedMLP

matplotlib.use("TkAgg")

# matplotlib 自带的 DejaVu Sans 没有中文字形，中文会变成方块。
# Noto Sans CJK 在 Arch 上是 noto-cjk 包提供的，按优先级取第一个装了的。
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
# 负号用 ASCII，否则中文字体里的负号可能显示成方块。
plt.rcParams["axes.unicode_minus"] = False


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run the loader once and return (confidence, predicted, labels) as numpy."""
    model.eval()
    confidences = []
    predictions = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probabilities = torch.softmax(model(inputs), dim=1)
            confidence, predicted = probabilities.max(dim=1)

            confidences.append(confidence.cpu())
            predictions.append(predicted.cpu())
            labels_all.append(labels)

    return (
        torch.cat(confidences).numpy(),
        torch.cat(predictions).numpy(),
        torch.cat(labels_all).numpy(),
    )


def accuracy_of(confidence: np.ndarray, predicted: np.ndarray, labels: np.ndarray):
    """Return (accuracy, mean confidence)."""
    return float((predicted == labels).mean()), float(confidence.mean())


def train_model(
    group: str,
    device: torch.device,
    data_root: str = "../../data",
    epochs: int = 10,
    batch_size: int = 256,
    hidden_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    p_a: float = 0.5,
):
    """Train one MLP on a single group. Returns (model, history).

    history holds one dict per epoch: loss, accuracy, confidence.
    """
    print(f"\n{'=' * 60}")
    print(f"training on group {group}")
    print(f"{'=' * 60}")

    torch.manual_seed(seed)

    train_set = HintedMNIST(data_root, train=True, group=group, p_a=p_a)
    test_set = HintedMNIST(data_root, train=False, group=group, seed=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    model = HintedMLP(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        confidence, predicted, labels = predict(model, test_loader, device)
        accuracy, mean_confidence = accuracy_of(confidence, predicted, labels)

        history.append({
            "loss": average_loss,
            "accuracy": accuracy,
            "confidence": mean_confidence,
        })
        print(f"  epoch {epoch:2d}/{epochs}  loss {average_loss:.4f}  "
              f"test acc {accuracy:.4f}  conf {mean_confidence:.4f}")

    return model, history


def plot_history(histories: dict[str, list[dict]], title: str = "training curves"):
    """One figure, two panels: accuracy and confidence over epochs."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    for group, history in histories.items():
        epochs = range(1, len(history) + 1)
        axes[0].plot(epochs, [row["accuracy"] for row in history], marker="o", label=f"group {group}")
        axes[1].plot(epochs, [row["confidence"] for row in history], marker="o", label=f"group {group}")

    axes[0].set_title("test accuracy")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("mean confidence")
    axes[1].set_xlabel("epoch")

    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_digits(
    model: nn.Module,
    dataset: HintedMNIST,
    device: torch.device,
    count: int = 16,
    only_mistakes: bool = False,
    seed: int = 0,
    title: str | None = None,
):
    """Draw `count` digits with their prediction, confidence and true label.

    Each title shows: 预测值 / 置信度, then the true digit, and the hint value.
    Wrong predictions get a red border, right ones a green border.
    """
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count * 8 if only_mistakes else count]

    columns = 4
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.4, rows * 2.8))
    axes = np.atleast_1d(axes).ravel()

    drawn = 0
    for index in indices:
        inputs, label = dataset[int(index)]
        with torch.no_grad():
            probabilities = torch.softmax(model(inputs.unsqueeze(0).to(device)), dim=1)
            confidence, predicted = probabilities.max(dim=1)

        predicted = int(predicted.item())
        confidence = float(confidence.item())
        label = int(label)

        if only_mistakes and predicted == label:
            continue
        if drawn >= count:
            break

        image = inputs[:784].reshape(28, 28).numpy()
        hint = inputs[784:]
        hint_value = "无提示" if float(hint.sum()) == 0 else int(hint.argmax())

        axis = axes[drawn]
        axis.imshow(image, cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])
        is_right = predicted == label
        # 单行短标签，避免换行后和相邻子图叠在一起。
        axis.set_title(
            f"{predicted} ({confidence:.2f}) 真{label} 提{hint_value}",
            fontsize=9,
            color="green" if is_right else "red",
            pad=4,
        )
        for spine in axis.spines.values():
            spine.set_edgecolor("green" if is_right else "red")
            spine.set_linewidth(2 if is_right else 3)

        drawn += 1

    for axis in axes[drawn:]:
        axis.axis("off")

    figure.suptitle(title or f"random {drawn} digits from the test set")
    figure.tight_layout()
    return figure


def plot_reliability(
    confidence: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
    title: str = "reliability curve",
):
    """Bucket by confidence, then compare each bucket's accuracy to its confidence.

    A well-calibrated model sits on the diagonal. Points *below* the diagonal
    mean "wrong but confident", which is the failure mode this project hunts.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    bucket = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    accuracy_per_bin = []
    confidence_per_bin = []
    counts = []

    for index in range(bins):
        mask = bucket == index
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            accuracy_per_bin.append(np.nan)
            confidence_per_bin.append(centers[index])
            continue
        accuracy_per_bin.append(float((predicted[mask] == labels[mask]).mean()))
        confidence_per_bin.append(float(confidence[mask].mean()))

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    left.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    left.plot(confidence_per_bin, accuracy_per_bin, "o-", label="model")
    left.set_xlabel("mean confidence in bucket")
    left.set_ylabel("accuracy in bucket")
    left.set_title(title)
    left.set_xlim(0, 1.02)
    left.set_ylim(0, 1.02)
    left.grid(alpha=0.3)
    left.legend()

    right.bar(centers, counts, width=1 / bins * 0.9)
    right.set_xlabel("confidence bucket")
    right.set_ylabel("sample count")
    right.set_title("bucket sizes")
    right.grid(alpha=0.3)

    figure.tight_layout()
    return figure


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    models = {}
    histories = {}
    for group in ("A", "C"):
        models[group], histories[group] = train_model(group, device, epochs=10)

    plot_history(histories)

    # Group C is the interesting one: the hint lies, so look at what it guesses.
    test_c = HintedMNIST("../../data", train=False, group="C", seed=1)
    plot_digits(models["C"], test_c, device, count=16,
                title="group C: hint contradicts the image")

    confidence, predicted, labels = predict(
        models["C"], DataLoader(test_c, batch_size=512), device
    )
    plot_reliability(confidence, predicted, labels, title="group C calibration")

    accuracy, mean_confidence = accuracy_of(confidence, predicted, labels)
    print(f"\ngroup C: acc {accuracy:.4f}  conf {mean_confidence:.4f}")
    print(f"overconfidence gap (conf - acc): {mean_confidence - accuracy:+.4f}")

    plt.show()


if __name__ == "__main__":
    main()
