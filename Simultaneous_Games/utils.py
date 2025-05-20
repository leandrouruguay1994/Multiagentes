# cell 1: Importaciones
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
import os
import time

# cell 2: Función para simular juegos
def simulate_game(game_class, agent_classes, n_iter=1000, initial_policies=None, seeds=None):
    """
    Simula un juego entre dos agentes y registra políticas/recompensas.
    """
    game = game_class()
    agents = [
        agent_classes[0](game, game.agents[0], initial=initial_policies[0], seed=seeds[0]),
        agent_classes[1](game, game.agents[1], initial=initial_policies[1], seed=seeds[1])
    ]
    
    # Historial de políticas y recompensas
    history = {
        'policies': {agent.agent: [] for agent in agents},
        'rewards': {agent.agent: [] for agent in agents}
    }
    
    for _ in tqdm(range(n_iter)):
        game.reset()
        actions = {agent.agent: agent.action() for agent in agents}
        _, rewards, _, _, _ = game.step(actions)
        
        # Guardar políticas y recompensas
        for agent in agents:
            history['policies'][agent.agent].append(agent.policy().copy())
            history['rewards'][agent.agent].append(rewards[agent.agent])
    
    return history

# cell 3: Función de visualización
def plot_results(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de políticas
    for agent, policies in history['policies'].items():
        policies = np.array(policies)
        for action in range(policies.shape[1]):
            ax1.plot(policies[:, action], label=f"{agent} - Acción {action}")
    ax1.set_title(f"Políticas: {title}")
    ax1.legend()
    
    # Gráfico de recompensas acumuladas
    for agent, rewards in history['rewards'].items():
        ax2.plot(np.cumsum(rewards), label=agent)
    ax2.set_title(f"Recompensas acumuladas: {title}")
    ax2.legend()
    
    plt.show()

def plot_training_results(stats, agents, game, save_local=True):
    """Versión modificada que devuelve figuras para W&B"""
    figures = {}
    
    # Figura 1: Recompensas por episodio
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for agent in game.agents:
        rewards = np.array(stats['episode_rewards'][agent])
        ax1.plot(rewards, label=f"{agent} ({type(agents[agent]).__name__})", alpha=0.7)
    ax1.set_title("Rewards per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    figures["rewards_per_episode"] = fig1
    
    # Figura 2: Recompensas promedio por iteración
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for agent in game.agents:
        avg_rewards = stats['average_rewards'][agent]
        ax2.plot(avg_rewards, label=f"{agent} ({type(agents[agent]).__name__})", marker='o')
    ax2.set_title("Average Rewards per Iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Average Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    figures["avg_rewards_per_iteration"] = fig2
    
    # Figura 3: Tiempos de ejecución
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(stats['time_per_episode'], color='purple')
    ax3.set_title("Execution Time per Episode")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Time (seconds)")
    ax3.grid(True, alpha=0.3)
    figures["execution_times"] = fig3
    
    # Guardar localmente si se solicita
    if save_local:
        os.makedirs("training_plots", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for name, fig in figures.items():
            fig.savefig(f"training_plots/{name}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close('all')
    
    return figures