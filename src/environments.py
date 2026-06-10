"""Environment definitions for CSCN8020 Assignment 1.

The main environment is a deterministic Gymnasium-style GridWorld.  Rewards are
state-based and are received when the agent enters the next state.  Invalid
moves keep the agent in the same state and return the reward of that same state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for simple notebook viewing
    class _FallbackEnv:
        pass

    class _FallbackSpaces:
        class Discrete:
            def __init__(self, n: int):
                self.n = n

    class gym:  # type: ignore
        Env = _FallbackEnv

    spaces = _FallbackSpaces()  # type: ignore


State = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    next_state: State
    reward: float
    terminated: bool


class DeterministicGridWorldEnv(gym.Env):
    """A small deterministic gridworld with four movement actions.

    Actions:
        0 -> right, 1 -> down, 2 -> left, 3 -> up

    The environment follows the standard Gymnasium ``reset`` and ``step`` pattern,
    while also exposing model methods needed for dynamic programming.
    """

    metadata = {"render_modes": ["ansi"]}

    ACTIONS: Dict[int, State] = {
        0: (0, 1),    # right
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (-1, 0),   # up
    }
    ACTION_SYMBOLS: Dict[int, str] = {
        0: "→",
        1: "↓",
        2: "←",
        3: "↑",
    }
    ACTION_NAMES: Dict[int, str] = {
        0: "right",
        1: "down",
        2: "left",
        3: "up",
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        rewards: Dict[State, float],
        terminal_states: Optional[Iterable[State]] = None,
        start_state: State = (0, 0),
        max_steps: int = 100,
        terminal_display_value: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.rewards = dict(rewards)
        self.terminal_states = set(terminal_states or [])
        self.start_state = start_state
        self.max_steps = max_steps
        self.terminal_display_value = terminal_display_value

        self.observation_space = spaces.Discrete(rows * cols)
        self.action_space = spaces.Discrete(4)
        self.state = start_state
        self.steps_taken = 0

        for state in self.all_states():
            self.rewards.setdefault(state, 0.0)

    def all_states(self) -> List[State]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def state_to_index(self, state: State) -> int:
        return state[0] * self.cols + state[1]

    def index_to_state(self, index: int) -> State:
        return divmod(index, self.cols)

    def is_inside(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def transition_from(self, state: State, action: int) -> StepResult:
        """Return deterministic transition for a state-action pair."""
        if self.is_terminal(state):
            return StepResult(next_state=state, reward=0.0, terminated=True)

        dr, dc = self.ACTIONS[action]
        candidate = (state[0] + dr, state[1] + dc)
        next_state = candidate if self.is_inside(candidate) else state
        reward = float(self.rewards[next_state])
        terminated = self.is_terminal(next_state)
        return StepResult(next_state=next_state, reward=reward, terminated=terminated)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self.state = (options or {}).get("start_state", self.start_state)
        self.steps_taken = 0
        return self.state_to_index(self.state), {"state": self.state}

    def step(self, action: int):
        result = self.transition_from(self.state, int(action))
        self.state = result.next_state
        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps and not result.terminated
        return (
            self.state_to_index(self.state),
            result.reward,
            result.terminated,
            truncated,
            {"state": self.state},
        )

    def reward_grid(self) -> np.ndarray:
        grid = np.zeros((self.rows, self.cols), dtype=float)
        for state, reward in self.rewards.items():
            grid[state] = reward
        return grid

    def render_values(self, values: np.ndarray) -> np.ndarray:
        return np.round(values.reshape(self.rows, self.cols), 2)

    def render_policy(self, policy: np.ndarray) -> np.ndarray:
        symbols = np.empty((self.rows, self.cols), dtype=object)
        for state in self.all_states():
            idx = self.state_to_index(state)
            if self.is_terminal(state):
                symbols[state] = "G"
            else:
                symbols[state] = self.ACTION_SYMBOLS[int(policy[idx])]
        return symbols


def create_2x2_gridworld() -> DeterministicGridWorldEnv:
    rewards = {
        (0, 0): 5.0,
        (0, 1): 10.0,
        (1, 0): 1.0,
        (1, 1): 2.0,
    }
    return DeterministicGridWorldEnv(rows=2, cols=2, rewards=rewards, terminal_states=[])


def create_5x5_gridworld() -> DeterministicGridWorldEnv:
    goal = (4, 4)
    grey_states = {(2, 2), (3, 0), (0, 4)}
    rewards: Dict[State, float] = {}
    for r in range(5):
        for c in range(5):
            state = (r, c)
            if state == goal:
                rewards[state] = 10.0
            elif state in grey_states:
                rewards[state] = -5.0
            else:
                rewards[state] = -1.0
    return DeterministicGridWorldEnv(
        rows=5,
        cols=5,
        rewards=rewards,
        terminal_states={goal},
        start_state=(0, 0),
        max_steps=100,
        terminal_display_value=10.0,
    )
