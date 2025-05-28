import time
import matplotlib.pyplot as plt
import subprocess

"""def play_episode(game, agents, verbose=False, render=False):

    # Initialize the game
    game.reset()
    step_count = 0

    # Initialize each agent
    for agent in game.agents:
        agents[agent].reset()

    # Print initial observations if verbose is enabled
    if verbose:
        print(f"Step: {step_count}")
        for agent in game.agents:
            print(f"Agent {agent} observe: {game.observe(agent)}")

    # Initialize rewards for each agent
    cum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))

    # Render the game if required
    if render:
        game.render()
        time.sleep(0.5)

    while not game.done():
        step_count += 1

        # Get actions from each agent
        actions = {}
        states = {}
        for agent in game.agents:
            obs = game.observe(agent)
            state = tuple(obs.tolist())
            states[agent] = state
            if hasattr(agents[agent], "select_action"):
                actions[agent] = agents[agent].select_action(state)
            else:
                actions[agent] = agents[agent].action()

        print(f"Actions in step {step_count}: {actions}")

        # Perform the actions in the game
        game.step(actions)

        # Update the cum_rewards for each agent
        for agent in game.agents:
            cum_rewards[agent] += game.reward(agent)

        # Print actions, rewards and next state if verbose is enabled
        if verbose:
            print(f"Step: {step_count}")
            for agent in game.agents:
                print(f"Agent {agent} action: {actions[agent]} - {game.action_set[actions[agent]]}")
                print(f"Agent {agent} reward: {game.reward(agent)}")
                print(f"Agent {agent} observe: {game.observe(agent)}")

        if render:
            game.render()
            time.sleep(0.5)

        for agent in game.agents:
            opponent = [a for a in game.agents if a != agent][0]
            obs = game.observe(agent)
            next_state = tuple(obs.tolist())

            if hasattr(agents[agent], "update"):
                try:
                    agents[agent].update(
                        state=states[agent],
                        action_i=actions[agent],
                        action_j=actions[opponent],
                        reward=game.reward(agent),
                        next_state=next_state
                    )
                except TypeError:
                    # El agente no acepta esos argumentos (como DumbAgent)
                    agents[agent].update()

    return cum_rewards

def run(game, agents, episodes=1, verbose=False, render=False):
    sum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))
    for _ in range(episodes):
        cum_rewards = play_episode(game, agents, verbose=verbose, render=render)  
        for agent in game.agents:
            sum_rewards[agent] += cum_rewards[agent]
    if verbose:
        print(f"Average rewards over {episodes} episodes:")
        for agent in game.agents:
            print(f"Agent {agent}: {sum_rewards[agent] / episodes}")  
    return sum_rewards 

def train(game, agents, train_config, progress=False, verbose=False, render=False):
    iterations = train_config["iterations"]
    episodes = train_config["episodes"]
    average_rewards = dict(map(lambda agent: (agent, []), game.agents))
    for i in range(1, iterations+1):
        sum_rewards = run(game, agents, episodes=episodes, verbose=verbose, render=render)
        for agent in game.agents:
            average_rewards[agent].append(sum_rewards[agent] / episodes)
        if progress and (i % 10 == 0):
            print(f"Iteration {i} ({i * episodes} episodes)")
            for agent in game.agents:
                print(f"Agent {agent}: {average_rewards[agent][-1]}")
    if progress:
        print(f"Last average rewards over {iterations} iterations ({iterations * episodes} episodes):")
        for agent in game.agents:
            print(f"Agent {agent}: {average_rewards[agent][-1]}")
    return average_rewards
"""
"""def plot_average_rewards(average_rewards):
    for agent, rewards in average_rewards.items():
        plt.plot(rewards, label=agent)

    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards per Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def play_episode(game, agents, verbose=False, render=True):
    # 0) Inicializar el juego y observaciones previas
    game.reset()
    prev_states = {agent: game.observe(agent) for agent in game.agents}
    
    # 1) Reset de cada agente
    for agent in game.agents:
        agents[agent].reset()
    
    cum_rewards = {agent: 0.0 for agent in game.agents}
    if render:
        game.render()
        time.sleep(0.5)
    
    # 2) Loop de episodio
    while not game.done():
        # 2.1) Todos eligen acción
        actions = {agent: agents[agent].action() for agent in game.agents}
        
        # 2.2) Ejecutar paso
        observations, _, _, _, _ = game.step(actions)
        
        # 2.3) Acumular recompensas
        for agent in game.agents:
            cum_rewards[agent] += game.reward(agent)
        
        # 2.4) Llamar a update para cada agente:
        #     update(prev_state, propia_action, *opponent_actions, reward, next_state)
        for agent in game.agents:
            # Listado de oponentes (1 o 2 para juegos de 2 o 3 agentes)
            opponents = [a for a in game.agents if a != agent]
            opponent_actions = [actions[a] for a in opponents]
            
            # Llamada posicional, expandiendo opponent_actions
            agents[agent].update(
                prev_states[agent],
                actions[agent],
                *opponent_actions,
                game.reward(agent),
                observations[agent]
            )
            
            # Actualizar estado previo
            prev_states[agent] = observations[agent]
        
        # 2.5) Verbose / Render opcional
        if verbose:
            print(f"Step finalizada — acciones: {actions} — recompensas: {cum_rewards}")
        if render:
            game.render()
            time.sleep(0.5)
    
    return cum_rewards"""

def plot_average_rewards(average_rewards):
    for agent, rewards in average_rewards.items():
        plt.plot(rewards, label=agent)

    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards per Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def play_episode(game, agents, verbose=False, render=True):

    # Initialize the game
    game.reset()
    step_count = 0
    #print("Field:")
    #print(game.env.unwrapped.field)
    #for i, p in enumerate(game.env.unwrapped.players):
    #    print(f"Player {i}: {p.position}")
    # Verificá el tamaño del campo

    # Initialize each agent
    for agent in game.agents:
        agents[agent].reset()

    # Print initial observations if verbose is enabled
    if verbose:
        print(f"Step: {step_count}")
        for agent in game.agents:
            print(f"Agent {agent} observe: {game.observe(agent)}")

    # Initialize rewards for each agent
    cum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))

    # render the game if required
    if render:
        game.render()
        time.sleep(0.5)

    while not game.done():

        step_count += 1
        
        # Get actions from each agent
        actions = {}
        for agent in game.agents:
            actions[agent] = agents[agent].action()
        # Perform the actions in the game
        observations,_,_,_,_ = game.step(actions)
        
        # Update the cum_rewards for each agent
        for agent in game.agents:
            cum_rewards[agent] += game.reward(agent)

        # Print actions, rewards and next state if verbose is enabled
        if verbose:
            print(f"Step: {step_count}")
            for agent in game.agents:
                    print(f"Agent {agent} action: {actions[agent]} - {game.action_set[actions[agent]]}")
                    print(f"Agent {agent} reward: {game.reward(agent)}")
                    print(f"Agent {agent} observe: {game.observe(agent)}")
            
        if render:
            game.render()
            time.sleep(0.5)
    
        for agent in game.agents:
            # Update the agent with the last observation
            agents[agent].update()
    
    return cum_rewards

def run(game, agents, episodes=1, verbose=False, render=False):
    sum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))
    for episode in range(episodes):
        if verbose:
            print(f"Episode {episode + 1}/{episodes}")
        cum_rewards = play_episode(game, agents, verbose=verbose, render=render)  
        for agent in game.agents:
            sum_rewards[agent] += cum_rewards[agent]
    if verbose:
        print(f"Average rewards over {episodes} episodes:")
        for agent in game.agents:
            print(f"Agent {agent}: {sum_rewards[agent] / episodes}")  
    return sum_rewards  

def train(game, agents, train_config, progress=False, verbose=False, render=False):
    iterations = train_config["iterations"]
    episodes = train_config["episodes"]
    average_rewards = dict(map(lambda agent: (agent, []), game.agents))
    for i in range(1, iterations+1):
        sum_rewards = run(game, agents, episodes=episodes, verbose=verbose, render=render)
        for agent in game.agents:
            average_rewards[agent].append(sum_rewards[agent] / episodes)
        if progress and (i % 10 == 0):
            print(f"Iteration {i} ({i * episodes} episodes)")
            for agent in game.agents:
                print(f"Agent {agent}: {average_rewards[agent][-1]}")
    if progress:
        print(f"Last average rewards over {iterations} iterations ({iterations * episodes} episodes):")
        for agent in game.agents:
            print(f"Agent {agent}: {average_rewards[agent][-1]}")
    return average_rewards





"""
def run(game, agents, episodes=1, verbose=False, render=False):
    sum_rewards = dict(map(lambda agent: (agent, 0.0), game.agents))
    for _ in range(episodes):
        cum_rewards = play_episode(game, agents, verbose=verbose, render=render)  
        for agent in game.agents:
            sum_rewards[agent] += cum_rewards[agent]
    if verbose:
        print(f"Average rewards over {episodes} episodes:")
        for agent in game.agents:
            print(f"Agent {agent}: {sum_rewards[agent] / episodes}")  
    return sum_rewards     

def train(game, agents, train_config, progress=False, verbose=False, render=False):
    iterations = train_config["iterations"]
    episodes = train_config["episodes"]
    average_rewards = dict(map(lambda agent: (agent, []), game.agents))
    for i in range(1, iterations+1):
        sum_rewards = run(game, agents, episodes=episodes, verbose=verbose, render=render)
        for agent in game.agents:
            average_rewards[agent].append(sum_rewards[agent] / episodes)
        if progress and (i % 10 == 0):
            print(f"Iteration {i} ({i * episodes} episodes)")
            for agent in game.agents:
                print(f"Agent {agent}: {average_rewards[agent][-1]}")
    if progress:
        print(f"Last average rewards over {iterations} iterations ({iterations * episodes} episodes):")
        for agent in game.agents:
            print(f"Agent {agent}: {average_rewards[agent][-1]}")
    return average_rewards
"""
def export_notebook_to_html(notebook_path, output_path):
    command = [
        "python", "-m", "jupyter", "nbconvert",
        "--to", "html",
        notebook_path,
        "--output", output_path
    ]
    subprocess.run(command, check=True)