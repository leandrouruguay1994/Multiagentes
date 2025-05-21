from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        self.count: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random 
        #       
        if initial is None:
            for agent in self.game.agents:
                self.count[agent] = np.random.randint(low=1, high=10, size = self.game.num_actions(agent))+1
        else:
            for agent in self.game.agents:
                self.count[agent] = initial[agent]
        self.learned_policy: dict[agent, ndarray] = {}
        # 
        # TODO: inicializar learned_policy usando de count
        # 
        #agents_actions = list(map(lambda agent: list(game.action_iter(agent)), game.agents))
        #total_count = np.sum(list(self.count.values()))
        for agent in self.game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])
        self.learn = True  # Para controlar si el agente está aprendiendo o no

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta 
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #
        #print(f"Product_agents_actions: {product(*agents_actions)}")
        #action_combinations = product(*[range(self.game.num_actions(agent)) for agent in self.game.agents])
        
        for actions in product(*agents_actions):
            # Crear un nuevo juego para cada combinación de acciones
            g = self.game.clone()
            action_dict = {agent: act for agent, act in zip(self.game.agents, actions)}
            g.step(action_dict)
            rewards[actions] = g.reward(self.agent)
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente. 
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        # Para cada acción posible del agente
        for my_action in range(self.game.num_actions(self.agent)):
            total = 0.0
            
            # Para cada combinación de acciones de los otros agentes
            for action_combination, reward in rewards.items():
                if action_combination[self.game.agent_name_mapping[self.agent]] == my_action:
                    # Calcular probabilidad conjunta de las acciones de los otros jugadores
                    prob = 1.0
                    for agent, action in zip(self.game.agents, action_combination):
                        if agent != self.agent:
                            prob *= self.learned_policy[agent][action]
                    
                    total += prob * reward
            
            utility[my_action] = total
        return utility
    
    def bestresponse(self):
        a = None
        #
        # TODO: retornar la acción de mayor utilidad
        #
        utilities = self.get_utility()
        a = np.argmax(utilities)
        return a
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if self.learn is False:
            return
        # Si actions es None, significa que el juego no ha comenzado o el agente no tiene acciones
        # disponibles. En este caso, no actualizamos la política.
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    