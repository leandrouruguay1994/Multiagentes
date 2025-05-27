import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import itertools
import random


@dataclass
class JALAgentConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    min_epsilon: float = 0.05
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.997
    initial_q: float = 0.0
    max_t: int = 10000
    seed: Optional[int] = None

class JALAgent:
    def __init__(self, game: SimultaneousGame, agent: AgentID, config: JALAgentConfig):
        self.game = game
        self.agent = agent
        self.config = config
        self.actions = list(self.game.action_iter(self.agent))
        self.epsilon = config.initial_epsilon
        self.q = defaultdict(lambda: np.full((len(self.actions), len(self.actions)), config.initial_q))
        self.opponent_policy = defaultdict(lambda: np.ones(len(self.actions)) / len(self.actions))
        self.opponent_counts = defaultdict(lambda: np.zeros(len(self.actions)))
        self.t = 0
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)

    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.t = 0
        self.opponent_policy.clear()
        self.opponent_counts.clear()
        self.q.clear()
        for state in self.game.states():
            self.q[state] = np.full((len(self.actions), len(self.actions)), self.config.initial_q)
            self.opponent_policy[state] = np.ones(len(self.actions)) / len(self.actions)
            self.opponent_counts[state] = np.zeros(len(self.actions))

    def select_action(self, state):
        self.t += 1
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        q_vals = [self.expected_q(state, a_i) for a_i in self.actions]
        best_action = np.argwhere(q_vals == np.max(q_vals)).flatten()
        return self.actions[np.random.choice(best_action)]

    def expected_q(self, state, a_i):
        idx_i = self.actions.index(a_i)
        probs = self.opponent_policy[state]
        return sum(probs[idx_j] * self.q[state][idx_i][idx_j] for idx_j in range(len(self.actions)))

    def update(self, state, action_i, action_j, reward, next_state):
        idx_i = self.actions.index(action_i)
        idx_j = self.actions.index(action_j)

        self.opponent_counts[state][idx_j] += 1
        self.opponent_policy[state] = self.opponent_counts[state] / self.opponent_counts[state].sum()

        best_q = max(self.expected_q(next_state, a) for a in self.actions)
        td_target = reward + self.config.gamma * best_q
        td_error = td_target - self.q[state][idx_i][idx_j]
        self.q[state][idx_i][idx_j] += self.config.alpha * td_error

        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def action(self):
        obs = self.game.observe(self.agent)
        state = tuple(obs.tolist())
        return self.select_action(state)

    def policy(self):
        return {a: self.expected_q(state, a) for a in self.actions}

    def observe_opponent_action(self, state, opponent_action):
        idx_j = self.actions.index(opponent_action)
        self.opponent_counts[state][idx_j] += 1
        self.opponent_policy[state] = self.opponent_counts[state] / self.opponent_counts[state].sum()
    
    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.t = 0
        self.epsilon = self.config.initial_epsilon
        self.opponent_policy.clear()
        self.opponent_counts.clear()
        self.q.clear()