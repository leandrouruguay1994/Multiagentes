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

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta 
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        #
        agents_actions_dict = {f"agent_{agent}": actions[agent] for agent,actions in enumerate(agents_actions)}
        #print(f"Product_agents_actions: {product(*agents_actions)}")
        g.step(agents_actions_dict)
        for actions in product(*agents_actions): # (A1,A2,..,Am) donde Ai es la accion del agente i, todas las combinaciones de acciones.
            #print(actions)
            if actions in product(*agents_actions):
                rewards[actions] = g.reward(self.agent)
            else:
                rewards[actions] = 0
        print(rewards)
        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente. 
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        for i in range(len(utility)):
            rewards_action = {action: reward for action, reward in rewards.items() if action[self.game.agent_name_mapping[self.agent]]==i }
            p_joint_actions = np.prod([self.learned_policy[agent][i] for agent in self.game.agents])
            utility[i] += p_joint_actions*np.sum(list(rewards_action.values()))
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
    