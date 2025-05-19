import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

@dataclass
class IQLAgentConfig:
    alpha: float = 0.1          # Learning rate
    gamma: float = 0.99         # Discount factor
    min_epsilon: float = 0.01   # Minimum exploration rate
    initial_epsilon: float = 1.0 # Initial exploration rate
    epsilon_decay: float = 0.995 # Epsilon decay rate
    initial_q: float = 0.0      # Initial Q-values
    max_t: int = 1000           # Maximum timesteps (puede usarse para ajustar epsilon decay)
    seed: Optional[int] = None            # Random seed

class IQLAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, config: IQLAgentConfig = None):
        super().__init__(game=game, agent=agent)
        
        # Configuración por defecto si no se proporciona
        if config is None:
            config = IQLAgentConfig()
        
        # Parámetros de aprendizaje
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.initial_epsilon
        self.epsilon_min = config.min_epsilon
        self.epsilon_decay = config.epsilon_decay
        self.initial_q = config.initial_q
        self.max_t = config.max_t
        self.learn = True  # Para controlar si el agente está aprendiendo o no
        
        # Tabla Q (usamos defaultdict para estados no vistos)
        self.q_table = {}
        
        # Estado actual y última acción
        self.current_state = None
        self.last_action = None
        self.timestep = 0
        
        # Para compatibilidad con tu interfaz
        self.curr_policy = np.ones(self.game.num_actions(self.agent)) / self.game.num_actions(self.agent)
        self.learned_policy = self.curr_policy.copy()
        
        np.random.seed(config.seed)
    
    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.current_state = self.game.observe(self.agent)
        current_state_key = self.state_to_key(self.current_state)
        if current_state_key not in self.q_table:
            # Inicializa la Q-table para el nuevo estado
            self.q_table[current_state_key] = np.full(self.game.num_actions(self.agent), self.initial_q)
        self.last_action = None
    
    def state_to_key(self, state):
        """Convierte el estado observado en una clave hashable para la Q-table"""
        if state is None:
            return None
        
        # Adapta esto según la estructura exacta de tus observaciones
        # if isinstance(state, dict):
        #     # Asumiendo que el estado tiene una parte de observación y energía
        #     obs = tuple(state['observation'].flatten())
        #     return obs
        if isinstance(state, np.ndarray):
            return tuple(state.astype(np.int32))
        else:
            return tuple(state)
        
    
    def update(self) -> None:
        """Actualiza la Q-table basada en la última experiencia"""
        if not self.learn or self.current_state is None or self.last_action is None:
            return
        
        # Obtener el estado actual
        new_state = self.game.observe(self.agent)
        reward = self.game.reward(self.agent)  # Use safe reward processing
        done = self.game.done()
        self.timestep += 1
        
        state_key = self.state_to_key(self.current_state)

        # Obtener el valor Q actual
        current_q = self.q_table[state_key][self.last_action]

        new_state_key = self.state_to_key(new_state)


        if new_state_key not in self.q_table:
            # Inicializa la Q-table para el nuevo estado
            self.q_table[new_state_key] = np.full(self.game.num_actions(self.agent), self.initial_q)
        
        
        # Calcular el target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[new_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Actualizar la Q-table
        self.q_table[state_key][self.last_action] = current_q + self.alpha * (target_q - current_q)
        
        # Actualizar la política basada en la Q-table
        #self.update_policy(state_key)
        
        # Decaimiento de epsilon (puede ser ajustado según max_t)
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon_min + (self.epsilon - self.epsilon_min) * self.epsilon_decay
        )
        
        # Actualizar el estado actual
        self.current_state = new_state
    
    def update_policy(self, state_key):
        """Actualiza la política basada en los valores Q actuales"""
        q_values = self.q_table[state_key]
        
        # Política epsilon-greedy

        #best_action = np.argmax(q_values)
        best_action = np.argwhere(q_values == np.max(q_values)).flatten()
        self.curr_policy = np.ones_like(q_values) * (self.epsilon / len(q_values))
        self.curr_policy[best_action] += (1 - self.epsilon)/ len(q_values)
        
        # Para learned_policy, usamos la política greedy basada en Q-values
        self.learned_policy = np.zeros_like(q_values)
        self.learned_policy[best_action] = 1.0
    
    def action(self):
        """Selecciona una acción según la política actual""" 
        self.current_state = self.game.observe(self.agent)
        state_key = self.state_to_key(self.current_state)
        
        # Actualizar la política para el estado actual
        if state_key in self.q_table:
            self.update_policy(state_key)
        else:
            # Estado no visto, usar política uniforme
            self.curr_policy = np.ones(self.game.num_actions(self.agent)) / self.game.num_actions(self.agent)
        
        # Seleccionar acción
        self.curr_policy /= np.sum(self.curr_policy)  # Normalización final
        self.last_action = np.random.choice(len(self.curr_policy), p=self.curr_policy)
        return self.last_action
    
    def policy(self):
        """Devuelve la política aprendida (greedy basada en Q-values)"""
        return self.learned_policy