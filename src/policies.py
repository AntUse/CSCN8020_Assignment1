"""Policy helper classes for CSCN8020 Assignment 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RandomPolicy:
    """Uniform random behavior policy."""

    n_actions: int = 4
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def action(self, _state_index: int) -> int:
        return int(self.rng.integers(self.n_actions))

    def probability(self, _state_index: int, _action: int) -> float:
        return 1.0 / self.n_actions


@dataclass
class GreedyPolicy:
    """Greedy policy with respect to a state-action value table."""

    q_values: np.ndarray

    def action(self, state_index: int) -> int:
        return int(np.argmax(self.q_values[state_index]))

    def probability(self, state_index: int, action: int) -> float:
        return 1.0 if action == self.action(state_index) else 0.0
