"""Agent implementations for dynamic programming and Monte Carlo methods."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .environments import DeterministicGridWorldEnv, State
from .policies import RandomPolicy


@dataclass
class ValueIterationResult:
    values: np.ndarray
    policy: np.ndarray
    iterations: int
    elapsed_seconds: float
    deltas: List[float]


@dataclass
class ValueIterationAgent:
    env: DeterministicGridWorldEnv
    gamma: float = 0.9
    theta: float = 1e-6
    max_iterations: int = 1000
    logger: logging.Logger | None = None

    def _q_value(self, values: np.ndarray, state: State, action: int) -> float:
        transition = self.env.transition_from(state, action)
        next_idx = self.env.state_to_index(transition.next_state)
        future = 0.0 if transition.terminated else self.gamma * values[next_idx]
        return transition.reward + future

    def greedy_policy(self, values: np.ndarray) -> np.ndarray:
        policy = np.zeros(self.env.rows * self.env.cols, dtype=int)
        for state in self.env.all_states():
            idx = self.env.state_to_index(state)
            if self.env.is_terminal(state):
                policy[idx] = 0
                continue
            q_values = [self._q_value(values, state, action) for action in range(4)]
            policy[idx] = int(np.argmax(q_values))
        return policy

    def run(self) -> ValueIterationResult:
        n_states = self.env.rows * self.env.cols
        values = np.zeros(n_states, dtype=float)
        deltas: List[float] = []
        start = time.perf_counter()

        if self.logger:
            self.logger.info("Starting synchronous value iteration: gamma=%s theta=%s", self.gamma, self.theta)

        for iteration in range(1, self.max_iterations + 1):
            new_values = values.copy()
            delta = 0.0
            for state in self.env.all_states():
                idx = self.env.state_to_index(state)
                if self.env.is_terminal(state):
                    display_value = self.env.terminal_display_value
                    new_values[idx] = 0.0 if display_value is None else display_value
                    continue
                q_values = [self._q_value(values, state, action) for action in range(4)]
                new_values[idx] = max(q_values)
                delta = max(delta, abs(new_values[idx] - values[idx]))
            values = new_values
            deltas.append(delta)
            if self.logger and (iteration <= 5 or iteration % 10 == 0):
                self.logger.info("Value Iteration iteration=%s delta=%.8f", iteration, delta)
            if delta < self.theta:
                break

        elapsed = time.perf_counter() - start
        policy = self.greedy_policy(values)
        if self.logger:
            self.logger.info("Finished synchronous value iteration in %s iterations and %.6f seconds", iteration, elapsed)
            self.logger.info("Final values: %s", np.round(values, 3).tolist())
        return ValueIterationResult(values=values, policy=policy, iterations=iteration, elapsed_seconds=elapsed, deltas=deltas)


@dataclass
class InPlaceValueIterationAgent(ValueIterationAgent):
    def run(self) -> ValueIterationResult:
        n_states = self.env.rows * self.env.cols
        values = np.zeros(n_states, dtype=float)
        deltas: List[float] = []
        start = time.perf_counter()

        if self.logger:
            self.logger.info("Starting in-place value iteration: gamma=%s theta=%s", self.gamma, self.theta)

        for iteration in range(1, self.max_iterations + 1):
            delta = 0.0
            for state in self.env.all_states():
                idx = self.env.state_to_index(state)
                old_value = values[idx]
                if self.env.is_terminal(state):
                    display_value = self.env.terminal_display_value
                    values[idx] = 0.0 if display_value is None else display_value
                    continue
                q_values = [self._q_value(values, state, action) for action in range(4)]
                values[idx] = max(q_values)
                delta = max(delta, abs(values[idx] - old_value))
            deltas.append(delta)
            if self.logger and (iteration <= 5 or iteration % 10 == 0):
                self.logger.info("In-place Value Iteration iteration=%s delta=%.8f", iteration, delta)
            if delta < self.theta:
                break

        elapsed = time.perf_counter() - start
        policy = self.greedy_policy(values)
        if self.logger:
            self.logger.info("Finished in-place value iteration in %s iterations and %.6f seconds", iteration, elapsed)
            self.logger.info("Final in-place values: %s", np.round(values, 3).tolist())
        return ValueIterationResult(values=values, policy=policy, iterations=iteration, elapsed_seconds=elapsed, deltas=deltas)


@dataclass
class MonteCarloResult:
    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    episodes: int
    elapsed_seconds: float
    visited_state_count: int


@dataclass
class OffPolicyMonteCarloAgent:
    env: DeterministicGridWorldEnv
    gamma: float = 0.9
    episodes: int = 10000
    max_steps_per_episode: int = 100
    seed: int = 42
    logger: logging.Logger | None = None
    q_values: np.ndarray = field(init=False)
    cumulative_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_states = self.env.rows * self.env.cols
        self.q_values = np.zeros((n_states, 4), dtype=float)
        self.cumulative_weights = np.zeros((n_states, 4), dtype=float)
        self.behavior_policy = RandomPolicy(n_actions=4, seed=self.seed)
        self.rng = np.random.default_rng(self.seed)

    def greedy_action(self, state_index: int) -> int:
        return int(np.argmax(self.q_values[state_index]))

    def generate_episode(self) -> List[Tuple[int, int, float]]:
        non_terminal_states = [s for s in self.env.all_states() if not self.env.is_terminal(s)]
        start_state = non_terminal_states[int(self.rng.integers(len(non_terminal_states)))]
        state_index, _ = self.env.reset(options={"start_state": start_state})
        episode: List[Tuple[int, int, float]] = []

        for _ in range(self.max_steps_per_episode):
            action = self.behavior_policy.action(state_index)
            next_state_index, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state_index, action, reward))
            state_index = next_state_index
            if terminated or truncated:
                break
        return episode

    def run(self) -> MonteCarloResult:
        start = time.perf_counter()
        if self.logger:
            self.logger.info("Starting off-policy Monte Carlo: episodes=%s gamma=%s", self.episodes, self.gamma)

        for episode_number in range(1, self.episodes + 1):
            episode = self.generate_episode()
            G = 0.0
            W = 1.0
            for state_index, action, reward in reversed(episode):
                G = self.gamma * G + reward
                self.cumulative_weights[state_index, action] += W
                self.q_values[state_index, action] += (
                    W / self.cumulative_weights[state_index, action]
                ) * (G - self.q_values[state_index, action])

                greedy = self.greedy_action(state_index)
                if action != greedy:
                    break
                behavior_prob = self.behavior_policy.probability(state_index, action)
                W = W / behavior_prob

            if self.logger and episode_number in {1, 10, 100, 1000, self.episodes}:
                self.logger.info("MC episode=%s length=%s", episode_number, len(episode))

        elapsed = time.perf_counter() - start
        values = np.max(self.q_values, axis=1)
        for state in self.env.terminal_states:
            idx = self.env.state_to_index(state)
            values[idx] = 0.0 if self.env.terminal_display_value is None else self.env.terminal_display_value
        policy = np.argmax(self.q_values, axis=1)
        visited_state_count = int(np.sum(np.max(self.cumulative_weights, axis=1) > 0))

        if self.logger:
            self.logger.info("Finished off-policy MC in %.6f seconds", elapsed)
            self.logger.info("MC estimated values: %s", np.round(values, 3).tolist())
        return MonteCarloResult(
            values=values,
            q_values=self.q_values.copy(),
            policy=policy,
            episodes=self.episodes,
            elapsed_seconds=elapsed,
            visited_state_count=visited_state_count,
        )
