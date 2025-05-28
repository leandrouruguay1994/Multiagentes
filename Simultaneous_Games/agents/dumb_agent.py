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

    def _has_adjacent_food(self):
        obs = self.game.observe(self.agent)
        if obs is None:
            raise ValueError("El agente no tiene una observación válida.")
        if len(obs) < 6:
            raise ValueError("La observación no tiene la longitud esperada.")
        
        # Obtener la posición de la comida y el agente
        fx,fy = obs[:2]  #posicion de la manzana
        fz = obs[2]  #estado de la manzana
        r,c = obs[3:5]  #posicion del agente
        z = obs[5]  #estado del agente
        # Chequeo de adyacencia
        if (abs(fx - r) + abs(fy - c)) == 1:
            # La comida está adyacente al agente
            # El agente puede cargar la comida
            return True
        return False



    def action(self):
        actions = np.array(self.game.action_iter(self.agent))
        if self.current_state is None:
            raise ValueError("El agente no ha sido reseteado o no tiene un estado actual.")
        if self.game.done():
            raise ValueError("El juego ha terminado. No se puede seleccionar una acción.")
        # Intentar hacer LOAD solo si hay comida adyacente
        self.counter += 1
        if self._has_adjacent_food():
            self.last_action = actions[5]  # LOAD
            self.counter = 0
        else:
            if self.counter <= 3:
                self.last_action = actions[2]  # SOUTH
            else:
                self.last_action = actions[4]  # EAST
        return self.last_action

    
    def policy(self):
        return self._policy
    
    def reset(self):
        """Resetea el estado del agente para un nuevo episodio"""
        self.current_state = self.game.observe(self.agent)
        self.counter = 0
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