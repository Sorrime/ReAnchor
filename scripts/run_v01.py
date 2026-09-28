"""v0.1: does an unstated rule get filled in confidently?

Run it, look at the table, then change one thing and run it again.

    .venv/bin/python scripts/run_v01.py
    .venv/bin/python scripts/run_v01.py --mask-prob 0.0
    .venv/bin/python scripts/run_v01.py --coverage 0.3 --steps 12000
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reanchor.env import World, WorldConfig, make_world, test_pairs, train_pairs
from reanchor.model import PriorModel


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_batch(
    world: World,
    contexts: torch.Tensor,
    predicates: torch.Tensor,
    batch_size: int,
    mask_prob: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample training pairs, hiding some stated values to teach the unstated case."""
    index = torch.randint(contexts.numel(), (batch_size,), generator=generator)
    context = contexts[index]
    predicate = predicates[index]

    label = world.value[context, predicate]
    target = (label > 0).long()

    keep = (torch.rand(batch_size, generator=generator) >= mask_prob).float()
    return context, predicate, keep, label * keep, target


def train(world: World, model: PriorModel, args: argparse.Namespace, device: torch.device) -> None:
    contexts, predicates = train_pairs(world)
    generator = torch.Generator().manual_seed(args.seed + 1)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, args.steps + 1):
        context, predicate, stated, value, target = sample_batch(
            world, contexts, predicates, args.batch_size, args.mask_prob, generator
        )
        logits = model(
            context.to(device), predicate.to(device), stated.to(device), value.to(device)
        )
        loss = loss_fn(logits, target.to(device))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:6d}   loss {loss.item():.4f}")


@torch.no_grad()
def predict_unstated(
    model: PriorModel, world: World, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    context, predicate = test_pairs(world)
    stated = torch.zeros(context.numel(), device=device)
    value = torch.zeros(context.numel(), device=device)
    logits = model(context.to(device), predicate.to(device), stated, value)
    probability = torch.softmax(logits, dim=-1)[:, 1].cpu()
    truth = (world.value[context, predicate] > 0).long()
    prior = world.prior[predicate]
    return probability, truth, prior


def binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    p = probability.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log2() + (1 - p) * (1 - p).log2())


def sweep_table(
    probability: torch.Tensor, truth: torch.Tensor, prior: torch.Tensor, n_bins: int
) -> list[dict[str, float]]:
    """Group test pairs by how strong the cross-context prior is."""
    strength = torch.maximum(prior, 1 - prior)
    correct = ((probability > 0.5).long() == truth).float()
    confidence = torch.maximum(probability, 1 - probability)

    edges = torch.linspace(0.5, float(strength.max()), n_bins + 1).tolist()
    rows: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (strength >= low) & (strength < high)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{low:.2f}-{high:.2f}",
                "n": float(mask.sum()),
                "prior": float(torch.maximum(prior[mask], 1 - prior[mask]).mean()),
                "acc": float(correct[mask].mean()),
                "conf": float(confidence[mask].mean()),
                "entropy": float(binary_entropy(probability[mask]).mean()),
            }
        )
    return rows


def expected_calibration_error(
    probability: torch.Tensor, truth: torch.Tensor, n_bins: int = 15
) -> float:
    confidence = torch.maximum(probability, 1 - probability)
    correct = ((probability > 0.5).long() == truth).float()
    edges = torch.linspace(0.5, 1.0, n_bins + 1)
    total = confidence.numel()
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() == 0:
            continue
        gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        error += gap * float(mask.sum()) / total
    return error


def render(rows: list[dict[str, float]], ece: float, n_test: int) -> None:
    print()
    print(f"  test pairs (unstated in context {11}): {n_test}   ECE: {ece:.4f}")
    print()
    print("  prior   n    acc    conf   entropy   |  optimal acc   conf   entropy")
    print("  " + "-" * 74)
    for row in rows:
        optimal_entropy = float(binary_entropy(torch.tensor([row["prior"]])))
        print(
            f"  {row['prior']:.2f}  {int(row['n']):4d}"
            f"  {row['acc']:.3f}  {row['conf']:.3f}  {row['entropy']:.3f}"
            f"   |  {row['prior']:.3f}      {row['prior']:.3f}   {optimal_entropy:.3f}"
        )
    print()
    print("  'optimal' is the Bayes reference: confidence should equal prior strength.")
    print("  conf >> prior  ->  the gap was filled in without carrying uncertainty.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-contexts", type=int, default=12)
    parser.add_argument("--n-predicates", type=int, default=200)
    parser.add_argument("--coverage", type=float, default=0.7)
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="v01")
    parser.add_argument("--out", default="results")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    world_config = WorldConfig(
        n_contexts=args.n_contexts,
        n_predicates=args.n_predicates,
        coverage=args.coverage,
        test_context=args.n_contexts - 1,
        seed=args.seed,
    )
    world = make_world(world_config)
    model = PriorModel(
        world.n_contexts, world.n_predicates, hidden=args.hidden, n_layers=args.n_layers
    ).to(device)

    n_train = int(world.stated.sum())
    print(f"device {device}   train pairs {n_train}   mask_prob {args.mask_prob}")
    train(world, model, args, device)

    probability, truth, prior = predict_unstated(model, world, device)
    rows = sweep_table(probability, truth, prior, args.n_bins)
    ece = expected_calibration_error(probability, truth)
    render(rows, ece, probability.numel())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"sweep_{args.tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {csv_path}")


if __name__ == "__main__":
    main()
