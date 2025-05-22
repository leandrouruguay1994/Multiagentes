import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import itertools

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
        agents_actions = [list(game.action_iter(agent)) for agent in game.agents]
        self.current_state = None
        self.joint_actions = list(itertools.product(*agents_actions))
        self.timestep = 0
        self.learn = True
        self.last_action = None


        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.current_state = self.game.observe(self.agent)
        self.last_action = None
    
    def state_to_key(self, state):
        """Convierte el estado observado en una clave hashable para la Q-table"""
        if state is None:
            raise ValueError("El estado no puede ser None.")
     
        return tuple(state.astype(np.int32))
    
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
    
    def get_true_opponent_policies(self, state):
        """Obtiene las políticas REALES de los oponentes (en la práctica, esto requiere acceso a sus modelos internos)."""
        policies = {}
        for opponent in self.game.agents:
            if opponent != self.agent:
                opponent_agent = None
                # Buscar el agente oponente en la lista de agentes
                for agent in self.game.agents:
                    if agent == opponent:
                        opponent_agent = agent
                        break
                
                if hasattr(opponent_agent, 'policy'):
                    policies[opponent] = opponent_agent.policy(state)
                else:
                    # Fallback: Asumir política uniforme 
                    policies[opponent] = {a: 1.0 / self.game.num_actions(opponent) 
                                        for a in range(self.game.num_actions(opponent))}
        return policies
    
    
    def get_exact_best_response(self, state):
        """Calcula la best response exacta asumiendo conocimiento de las políticas actuales de los oponentes."""
        state_key = self.state_to_key(state)
        my_actions = range(self.game.num_actions(self.agent))
        best_action = None
        max_q_value = -np.inf

        # Paso 1: Obtener políticas actuales de los oponentes (requiere acceso real o estimación exacta)
        opponent_policies = self.get_true_opponent_policies(state)  # Nuevo método crítico

        # Paso 2: Para cada acción propia, calcular el Q-value esperado dado las políticas de los oponentes
        for my_action in my_actions:
            expected_q = 0.0
            # Generar todas las posibles combinaciones de acciones oponentes
            opponent_actions = [range(self.game.num_actions(opponent)) 
                              for opponent in self.game.agents if opponent != self.agent]
            joint_opponent_actions = itertools.product(*opponent_actions)

            for opp_actions in joint_opponent_actions:
                # Construir la acción conjunta
                joint_action = {self.agent: my_action}
                joint_action.update({opp: a for opp, a in zip(
                    [a for a in self.game.agents if a != self.agent], opp_actions)})
                joint_action_key = tuple(sorted(joint_action.items()))

                # Calcular probabilidad de esta acción conjunta según políticas de oponentes
                prob = 1.0
                for opp, a in zip([a for a in self.game.agents if a != self.agent], opp_actions):
                    prob *= opponent_policies[opp].get(a, 0.0)

                # Acumular Q-value ponderado por probabilidad
                expected_q += self.q_table[(state_key, joint_action_key)] * prob

            # Actualizar best action si encontramos un mejor Q-value esperado
            if expected_q > max_q_value:
                max_q_value = expected_q
                best_action = my_action

        return best_action


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
        
        if np.random.random() < self.config.initial_epsilon and self.learn:
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
        self.last_action = chosen_action
        return chosen_action
    
    def update(self):
        """Actualiza Q-values usando la best response exacta en el target."""
        if not self.learn or self.current_state is None or self.last_joint_action is None:
            return
        
        new_state = self.game.observe(self.agent)
        reward = float(np.array(self.game.reward(self.agent)).item())  # Asegurar float
        done = self.game.done()
        state_key = self.state_to_key(self.current_state)
        new_state_key = self.state_to_key(new_state)
        self.timestep += 1
        
        # Actualizar modelo de oponentes
        joint_action = {self.agent: self.last_joint_action[0][1]}
        joint_action.update(dict(self.last_joint_action[1:]))
        self.update_opponent_models(joint_action)
        
        # Calcular target Q-value
        if done:
            target_q = reward
        else:

            best_action = self.get_exact_best_response(new_state)
            opponent_policies = self.get_true_opponent_policies(new_state)
            # Obtener mejor Q-value para el nuevo estado
            # Calcular max_next_q
            max_next_q = 0.0
            opponent_actions = [
                range(self.game.num_actions(opponent_id)) 
                for opponent_id in opponent_policies
            ]
            for opp_actions in itertools.product(*opponent_actions):
                joint_action = {self.agent: best_action}
                joint_action.update({
                    opponent_id: action 
                    for opponent_id, action in zip(opponent_policies.keys(), opp_actions)
                })
                prob = np.prod([
                    policy.get(action, 0.0)
                    for (opponent_id, action), policy in zip(
                        zip(opponent_policies.keys(), opp_actions),
                        opponent_policies.values()
                    )
                ])
                max_next_q += self.q_table.get(
                    (new_state_key, tuple(sorted(joint_action.items()))), 
                    0.0
                ) * prob
            
            target_q = reward + self.config.gamma * max_next_q
            current_q = self.q_table.get((state_key, self.last_joint_action), 0.0)
            self.q_table[(state_key, self.last_joint_action)] = current_q + self.config.alpha * (target_q - current_q)
            
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