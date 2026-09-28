"""A small model over (context, predicate) pairs.

Inputs at every step:

    context id | predicate id | stated flag | stated value

The stated flag is what makes "unstated" representable at all. Without it the
model cannot tell a missing rule from a rule it has never been asked about,
and the question in README section 3 becomes unanswerable rather than open.
"""

from __future__ import annotations

import torch
from torch import nn


class PriorModel(nn.Module):
    def __init__(
        self,
        n_contexts: int,
        n_predicates: int,
        d_context: int = 8,
        d_predicate: int = 32,
        hidden: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.context_embedding = nn.Embedding(n_contexts, d_context)
        self.predicate_embedding = nn.Embedding(n_predicates, d_predicate)

        width = d_context + d_predicate + 2
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        context: torch.Tensor,
        predicate: torch.Tensor,
        stated: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.context_embedding(context),
                self.predicate_embedding(predicate),
                stated.unsqueeze(-1),
                value.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp(features)
