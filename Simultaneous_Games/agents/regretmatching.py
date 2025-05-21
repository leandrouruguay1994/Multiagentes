import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict

class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if initial is None:
          self.curr_policy = np.ones(self.game.num_actions(self.agent)) / self.game.num_actions(self.agent)
        else:
          self.curr_policy = np.array(initial, dtype=float)
          self.curr_policy /= self.curr_policy.sum()  # Normalizar

        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        #self.sum_policy = self.curr_policy.copy()
        self.sum_policy = np.zeros(self.game.num_actions(self.agent))
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)
        self.learn = True  # Para controlar si el agente está aprendiendo o no

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()
        a = actions[self.agent]
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float) # la utilidad de cada accion mia. 
        # 
        # TODO: calcular regrets
        #
        for action in range(g.num_actions(self.agent)):
            g_sim = self.game.clone()
            actions_sim = actions.copy()
            actions_sim[self.agent] = action
            g_sim.step(actions_sim)
            u[action] = g_sim.reward(self.agent)

        current_u = u[a]
        #print(g.num_actions(self.agent))
        #print(type(g.num_actions(self.agent)))
        r = u - current_u
        return r
    
    def regret_matching(self):
        #
        # TODO: calcular curr_policy y actualizar sum_policy
        #
        actions = self.game.observe(self.agent)
        regrets = self.regrets(actions)
        regrets = np.array([np.maximum(regret, 0.0) for action, regret in enumerate(regrets)])
        
        total = regrets.sum()

        if total > 0:
            self.curr_policy = regrets / total
        else:
            self.curr_policy = np.ones_like(self.cum_regrets) / len(self.cum_regrets)

        self.sum_policy += self.curr_policy


    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if self.learn == False:
           return   
        if actions is None:
           return
        
        regrets = self.regrets(actions)
        self.cum_regrets += regrets

        self.regret_matching()

        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))
    
    def policy(self):
        return self.learned_policy
