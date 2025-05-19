import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class JALAgentConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    min_epsilon: float = 0.05
    initial_epsilon: float = 1.0
    epsilon_decay: float = 0.997
    initial_q: float = 0.0  # Como en el libro
    max_t: int = 10000
    seed: Optional[int] = None

class JALAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, config: JALAgentConfig = None):
        super().__init__(game=game, agent=agent)
        self.config = config if config else JALAgentConfig()
        
        # Tablas Q para acciones conjuntas (estado, acción conjunta) -> valor
        self.q_table = defaultdict(lambda: self.config.initial_q)
        
        # Modelo de otros agentes (frecuencias de acciones)
        self.opponent_models: Dict[AgentID, Dict[Tuple, Dict[int, int]]] = {
            aid: defaultdict(lambda: defaultdict(int))
            for aid in game.agents if aid != agent
        }
        
        # Historial de estados y acciones
        self.current_state = None
        self.last_joint_action = None
        self.timestep = 0
        self.learn = True
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.current_state = self.game.observe(self.agent)
        self.last_joint_action = None
    
    def state_to_key(self, state):
        """Convierte el estado a una tupla hashable"""
        if state is None:
            return "initial"
        
        if isinstance(state, np.ndarray):
            return tuple(state.astype(np.int32).flatten())
        elif isinstance(state, dict):
            return tuple(sorted((k, tuple(v) if isinstance(v, np.ndarray) else v) 
                        for k, v in state.items()))
        return tuple(state) if isinstance(state, (list, tuple)) else (state,)
    
    def update_opponent_models(self, joint_action: ActionDict):
        """Actualiza los modelos de otros agentes basado en sus acciones"""
        state_key = self.state_to_key(self.current_state)
        for agent_id, action in joint_action.items():
            if agent_id != self.agent:
                self.opponent_models[agent_id][state_key][action] += 1
    
    def get_opponent_action_probs(self, state):
        """Obtiene probabilidades de acción para otros agentes"""
        state_key = self.state_to_key(state)
        probs = {}
        
        for agent_id in self.opponent_models:
            total = sum(self.opponent_models[agent_id][state_key].values())
            if total > 0:
                probs[agent_id] = {
                    a: count/total 
                    for a, count in self.opponent_models[agent_id][state_key].items()
                }
            else:
                # Distribución uniforme si no hay datos
                num_actions = self.game.num_actions(agent_id)
                probs[agent_id] = {a: 1/num_actions for a in range(num_actions)}
        
        return probs
    
    def sample_opponent_actions(self, state):
        """Muestra acciones de otros agentes según sus distribuciones"""
        probs = self.get_opponent_action_probs(state)
        return {
            agent_id: np.random.choice(list(actions.keys()), p=list(actions.values()))
            for agent_id, actions in probs.items()
        }
    
    def get_joint_action_distribution(self, state):
        """Calcula distribución de probabilidad para acciones conjuntas"""
        state_key = self.state_to_key(state)
        my_actions = range(self.game.num_actions(self.agent))
        opponent_actions = self.sample_opponent_actions(state)
        
        joint_actions = []
        q_values = []
        
        for my_action in my_actions:
            joint_action = {self.agent: my_action}
            joint_action.update(opponent_actions)
            joint_action_key = tuple(sorted(joint_action.items()))
            
            joint_actions.append((my_action, joint_action_key))
            q_values.append(self.q_table[(state_key, joint_action_key)])
        
        # Softmax con temperatura basada en epsilon
        temperature = max(0.1, self.config.initial_epsilon)
        exp_q = np.exp(np.array(q_values) / temperature)
        softmax_probs = exp_q / exp_q.sum()
        
        return softmax_probs, joint_actions
    
    def action(self):
        """Selecciona una acción usando política epsilon-greedy sobre acciones conjuntas"""
        self.current_state = self.game.observe(self.agent)
        probs, joint_actions = self.get_joint_action_distribution(self.current_state)
        
        if np.random.random() < self.config.initial_epsilon:
            # Exploración: selección aleatoria basada en probabilidades conjuntas
            action_idx = np.random.choice(len(probs), p=probs)
        else:
            # Explotación: selección basada en Q-values (sin usar argmax directamente)
            q_values = [self.q_table[(self.state_to_key(self.current_state), ja)] 
                       for _, ja in joint_actions]
            max_q = max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            action_idx = np.random.choice(best_indices)
        
        chosen_action, self.last_joint_action = joint_actions[action_idx]
        return chosen_action
    
    def update(self):
        """Actualiza Q-values y modelos basados en la última experiencia"""
        if not self.learn or self.current_state is None or self.last_joint_action is None:
            return
        
        new_state = self.game.observe(self.agent)
        reward = float(np.array(self.game.reward(self.agent)).item())  # Asegurar float
        done = self.game.done()
        self.timestep += 1
        
        # Actualizar modelo de oponentes
        joint_action = {self.agent: self.last_joint_action[0][1]}
        joint_action.update(dict(self.last_joint_action[1:]))
        self.update_opponent_models(joint_action)
        
        state_key = self.state_to_key(self.current_state)
        joint_action_key = tuple(sorted(joint_action.items()))
        new_state_key = self.state_to_key(new_state)
        
        # Obtener valor Q actual
        current_q = self.q_table[(state_key, joint_action_key)]
        
        # Calcular target Q-value
        if done:
            target_q = reward
        else:
            # Obtener mejor Q-value para el nuevo estado
            max_next_q = -np.inf
            opponent_probs = self.get_opponent_action_probs(new_state)
            
            for my_action in range(self.game.num_actions(self.agent)):
                joint_action_next = {self.agent: my_action}
                for agent_id in opponent_probs:
                    # Tomar la acción más probable del oponente
                    best_opponent_action = max(opponent_probs[agent_id].items(), key=lambda x: x[1])[0]
                    joint_action_next[agent_id] = best_opponent_action
                
                joint_action_next_key = tuple(sorted(joint_action_next.items()))
                q_value = self.q_table[(new_state_key, joint_action_next_key)]
                if q_value > max_next_q:
                    max_next_q = q_value
            
            target_q = reward + self.config.gamma * max_next_q
        
        # Actualizar Q-value
        self.q_table[(state_key, joint_action_key)] += self.config.alpha * (target_q - current_q)
        
        # Decaimiento de epsilon
        self.config.initial_epsilon = max(
            self.config.min_epsilon,
            self.config.initial_epsilon * self.config.epsilon_decay
        )
        
        # Actualizar estado actual
        self.current_state = new_state
    
    def policy(self):
        """Devuelve la política aprendida (greedy basada en Q-values)"""
        if self.current_state is None:
            num_actions = self.game.num_actions(self.agent)
            return np.ones(num_actions) / num_actions
        
        state_key = self.state_to_key(self.current_state)
        my_actions = range(self.game.num_actions(self.agent))
        opponent_probs = self.get_opponent_action_probs(self.current_state)
        
        # Encontrar la mejor acción conjunta
        best_value = -np.inf
        best_actions = []
        
        for my_action in my_actions:
            joint_action = {self.agent: my_action}
            for agent_id in opponent_probs:
                best_opponent_action = max(opponent_probs[agent_id].items(), key=lambda x: x[1])[0]
                joint_action[agent_id] = best_opponent_action
            
            joint_action_key = tuple(sorted(joint_action.items()))
            q_value = self.q_table[(state_key, joint_action_key)]
            
            if q_value > best_value:
                best_value = q_value
                best_actions = [my_action]
            elif q_value == best_value:
                best_actions.append(my_action)
        
        # Crear política greedy
        policy = np.zeros(self.game.num_actions(self.agent))
        for a in best_actions:
            policy[a] = 1.0 / len(best_actions)
        
        return policy