# cell 1: Importaciones
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

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