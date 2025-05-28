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
        self.actions = np.array(self.game.action_iter(self.agent))
        self.epsilon = config.initial_epsilon
        self.q = defaultdict(lambda: defaultdict(float)) # tabla_q por estado y acción conjunta
        # Generar todas las combinaciones de acciones posibles para los oponentes
        actions_combinations = list(itertools.product(self.actions, repeat=len(self.game.agents)-1))
        #self.opponent_counts = defaultdict(lambda: np.zeros(len(self.game.agents),len(self.actions)))
        self.opponent_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(len(self.actions), dtype=int))) # opponent_counts por agente
        self.opponent_policy = defaultdict(lambda: defaultdict(lambda: np.full(len(self.actions), 1/len(self.actions), dtype=float))) # policy por agente

        self.t = 0
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
        self.last_state = None
        self.current_state = None
        self.last_action = None
        self.learn = True
        self.max_t = config.max_t

    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.t = 0
        self.current_state = self.game.observe(self.agent)
        #current_state_key = tuple(self.current_state.tolist())

    def action_values(self, state, action_i):
        av = 0.0
        idx_i = self.game.agent_name_mapping[self.agent]  # posición del agente i

        # Iterar sobre todas las combinaciones de acciones de los oponentes
        opponent_agents = [a for a in self.game.agents if a != self.agent]
        opponent_indices = [self.game.agent_name_mapping[a] for a in opponent_agents]

        # Obtener las políticas estimadas para cada oponente en este estado
        opponent_policies = [self.opponent_policy[state][a] for a in opponent_agents]

        # Generar el producto cartesiano de acciones de los oponentes
        from itertools import product
        joint_opponent_actions = product(*[range(len(self.actions)) for _ in opponent_agents])

        for joint_a_minus_i in joint_opponent_actions:
            # Probabilidad conjunta del joint action de los oponentes
            p = 1.0
            for pi, a_j in zip(opponent_policies, joint_a_minus_i):
                p *= pi[a_j]

            # Construir la joint action completa con la acción fija del agente i
            joint_action = [None] * len(self.game.agents)
            joint_action[idx_i] = action_i
            for idx, a_j in zip(opponent_indices, joint_a_minus_i):
                joint_action[idx] = a_j
            joint_action = tuple(joint_action)

            # Acumular valor Q ponderado
            av += p * self.q[state][joint_action]

        return av


    def update(self):
        self.last_state = self.current_state
        if not self.learn:
            self.current_state = self.game.observe(self.agent)
            return
        if self.current_state is None or self.last_action is None:
            raise ValueError("El estado actual es None. Asegúrate de que el agente ha sido reseteado correctamente.")
        
        observed_joint_action = self.game.observe_action(self.agent)# if agent != self.agent}
        observed_agent_action = observed_joint_action[self.game.agent_name_mapping[self.agent]]

        if observed_joint_action is None or observed_agent_action is None:
            raise ValueError("No se pudo observar la acción conjunta o la acción del agente.")
        reward = self.game.reward(self.agent)
        if reward is None:
            raise ValueError("No se pudo obtener la recompensa del juego.")
        
        last_state = tuple(self.last_state)
        next_state = self.game.observe(self.agent)
        


        for agent_idx, action in enumerate(observed_joint_action):
            agent_name = f"agent_{agent_idx}"  # Los agentes se llaman agent_0, agent_1, etc.
            action_int = int(action)  # Convertimos la acción a entero (0-5)
            
            # Incrementamos el contador para este agente en este estado
            #opponent_actions_tuple = tuple(self.opponent_counts[last_state][agent_name])
            if agent_name != self.agent:  # No contamos la acción del propio agente
                self.opponent_counts[last_state][agent_name][action_int] += 1

        for agent in self.game.agents:
            if agent == self.agent:
                continue
            # Actualizar la policy (normalizar los counts)
            total = np.sum(self.opponent_counts[last_state][agent])
            if total > 0:
                self.opponent_policy[last_state][agent] = (
                    self.opponent_counts[last_state][agent] / total
            )
  
        next_state_key = tuple(next_state.tolist())
        action_values = [self.action_values(next_state_key, a_i) for a_i in self.actions]
        #best_q = max(self.action_values(tuple(next_state), a) for a in self.game.action_iter(self.agent))
        best_q = max(action_values)
        td_target = reward + self.config.gamma * best_q
        td_error = td_target - self.q[last_state][observed_joint_action]
        self.q[last_state][observed_joint_action] += self.config.alpha * td_error

        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

        self.current_state = next_state
        self.t += 1

        if self.t >= self.max_t:
            self.learn = False
            print(f"Agente {self.agent} ha alcanzado el máximo de iteraciones: {self.max_t}. Dejará de aprender.")

    def action(self):
        current_state = self.current_state
        if current_state is None:
            raise ValueError("El agente no ha sido reseteado o no tiene un estado actual.")
        if self.game.done():
            raise ValueError("El juego ha terminado. No se puede seleccionar una acción.")
        state = tuple(current_state.tolist())
        return self.select_action(state)
    
    def select_action(self, state):
        self.t += 1
        if self.learn and (np.random.rand() < self.epsilon):
            self.last_action = random.choice(self.actions)
            return self.last_action
        q_vals = [self.action_values(state, a_i) for a_i in self.actions]
        best_action = np.argwhere(q_vals == np.max(q_vals)).flatten()
        self.last_action = self.actions[np.random.choice(best_action)]
        if self.last_action is None:
            raise ValueError("No se pudo seleccionar una acción válida.")
        return self.last_action

    def policy(self):
        return {a: self.action_values(self.current_state, a) for a in self.actions}

    def observe_opponent_action(self, state, opponent_action):
        idx_j = self.actions.index(opponent_action)
        self.opponent_counts[state][idx_j] += 1
        self.opponent_policy[state] = self.opponent_counts[state] / self.opponent_counts[state].sum()
