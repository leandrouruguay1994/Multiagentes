import tqdm
import numpy as np
from collections import defaultdict
import random
from typing import Dict, Tuple

class IQLAgent:
    def __init__(self, agent_id, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q: estado -> array de valores Q para cada acción
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        
    def state_to_key(self, state):
        """Convierte el estado observado en una clave hashable para la Q-table"""
        # Aquí necesitamos procesar el estado observado para convertirlo en una tupla
        # que pueda servir como clave en el diccionario.
        # Esto depende de la estructura exacta de tus observaciones.
        
        # Ejemplo básico (debes adaptarlo a tu estructura de observación):
        if isinstance(state, dict):
            # Asumiendo que el estado tiene una parte de observación y energía
            obs_part = tuple(state['observation'].flatten())
            energy_part = (state['energy'],)
            return obs_part + energy_part
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten())
        else:
            return tuple(state)
    
    def act(self, state):
        """Selecciona una acción usando política epsilon-greedy"""
        state_key = self.state_to_key(state)
        
        if random.random() < self.epsilon:
            return self.action_space.sample()  # Exploración aleatoria
        else:
            return np.argmax(self.q_table[state_key])  # Explotación
    
    def learn(self, state, action, reward, next_state, done):
        """Actualiza la Q-table usando la regla de Q-learning"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Actualización Q-learning
        self.q_table[state_key][action] = current_q + self.alpha * (target_q - current_q)
        
        # Decaimiento de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_tabular_iql(env, num_episodes=1000):
    """Función de entrenamiento para Independent Q-Learning tabular"""
    
    # Inicializar agentes
    agents = {agent: TabularIQLAgent(agent, env.action_spaces[agent]) 
              for agent in env.agents}
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        env.reset()
        episode_reward = {agent: 0 for agent in env.agents}
        
        for agent in env.agent_iter():
            state = env.observations[agent]
            action = agents[agent].act(state)
            
            # Ejecutar acción en el ambiente
            env.step({agent: action})
            
            # Obtener nueva observación y recompensa
            next_state = env.observations[agent]
            reward = env.rewards[agent]
            done = env.terminations[agent]
            
            # Aprender de la experiencia
            agents[agent].learn(state, action, reward, next_state, done)
            
            # Acumular recompensa
            episode_reward[agent] += reward
            
            if done:
                break
        
        # Registrar progreso
        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agents[env.agents[0]].epsilon:.2f}")
    
    return agents, episode_rewards

# Ejemplo de uso
if __name__ == "__main__":
    # Crear el entorno (usando tu implementación)
    foraging_game = Foraging(config="Foraging-8x8-2p-1f-v3")
    
    # Entrenar los agentes
    trained_agents, rewards = train_tabular_iql(foraging_game, num_episodes=1000)
    
    # Para usar los agentes entrenados:
    # state = env.reset()
    # action = trained_agents[agent_id].act(state)