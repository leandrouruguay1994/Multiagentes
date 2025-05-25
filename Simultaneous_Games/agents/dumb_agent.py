import numpy as np
from base.game import SimultaneousGame, AgentID
from base.agent import Agent

class DumbAgent(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        if initial is None:
            self._policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
            self._policy = initial
        self.timestep = 0
        self.current_state = None
        self.last_action = None
        self.learn = True
        self.counter = 0

    def action(self):
        actions = np.array(self.game.action_iter(self.agent))
        if self.counter == 3:
            action = actions[5]
            self.last_action = action
            self.counter = 0
        else:
            self.counter += 1
            action = actions[2]
            self.last_action = action
        return action
    
    def policy(self):
        return self._policy
    
    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.current_state = self.game.observe(self.agent)

        self.last_action = None

    def update(self) -> None:
        """Actualiza la política del agente"""
        if self.current_state is None or self.last_action is None:
            raise ValueError("El agente no ha sido reseteado o no tiene una acción anterior.")
        
        # Obtener el estado actual
        new_state = self.game.observe(self.agent)
        #reward = self.game.reward(self.agent)
        #done = self.game.done()
        #state_key = self.state_to_key(self.current_state)
        #new_state_key = self.state_to_key(new_state)
        self.timestep += 1
        self.current_state = new_state