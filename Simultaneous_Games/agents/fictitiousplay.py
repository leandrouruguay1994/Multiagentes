from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        self.count: dict[AgentID, ndarray] =  {}
        if initial is None:
            for agent in self.game.agents:
                self.count[agent] = np.random.random(game.num_actions(agent))
        else:
            for agent in self.game.agents:
                self.count[agent] = initial[agent]

        self.learned_policy: dict[AgentID, ndarray] = {}

        for agent in self.count


    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        joint_actions = list(product(*agents_actions))
        rewards: dict[tuple, float] = {joint_action: g.reward(joint_action) for joint_action in joint_actions}
        
        return rewards
    
    def get_utility(self):
        g = self.game.clone()
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        
        for index in range(self.game.num_actions(self.agent)):
            utility[index] = rewards[index]

        # TODO: calcular la utilidad (valor) de cada acción de agente. 
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        return utility
    
    def bestresponse(self):
        a = np.argmax(self.get_utility)
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
    