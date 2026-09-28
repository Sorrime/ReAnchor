"""Synthetic world: one sparse rule table per context, shared predicate pool.

Design notes (see README section 4.2 and 4.7):

  * context identity is given, the rule table is never given
  * every (context, predicate) pair is either *stated* (with a value) or
    *unstated*; the model never sees a per-context table, only sampled pairs
  * the test set is the set of pairs that are unstated in the test context
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WorldConfig:
    n_contexts: int = 12
    n_predicates: int = 200
    coverage: float = 0.7
    prior_low: float = 0.05
    prior_high: float = 0.95
    test_context: int = 11
    seed: int = 0


@dataclass
class World:
    prior: torch.Tensor
    stated: torch.Tensor
    value: torch.Tensor
    config: WorldConfig

    @property
    def n_contexts(self) -> int:
        return int(self.config.n_contexts)

    @property
    def n_predicates(self) -> int:
        return int(self.config.n_predicates)


def make_world(config: WorldConfig) -> World:
    generator = torch.Generator().manual_seed(config.seed)
    n_contexts, n_predicates = config.n_contexts, config.n_predicates

    prior = torch.rand(n_predicates, generator=generator)
    prior = config.prior_low + prior * (config.prior_high - config.prior_low)

    draw = torch.rand(n_contexts, n_predicates, generator=generator)
    value = torch.where(draw < prior.unsqueeze(0), 1.0, -1.0)

    stated = torch.rand(n_contexts, n_predicates, generator=generator) < config.coverage

    return World(prior=prior, stated=stated, value=value, config=config)


def train_pairs(world: World) -> tuple[torch.Tensor, torch.Tensor]:
    index = world.stated.nonzero()
    return index[:, 0].contiguous(), index[:, 1].contiguous()


def test_pairs(world: World) -> tuple[torch.Tensor, torch.Tensor]:
    test_context = world.config.test_context
    predicates = (~world.stated[test_context]).nonzero().squeeze(-1)
    contexts = torch.full_like(predicates, test_context)
    return contexts, predicates


def observed_contexts(world: World, predicate: int) -> torch.Tensor:
    """Contexts whose table states this predicate. Useful for sanity checks."""
    return world.stated[:, predicate].nonzero().squeeze(-1)
