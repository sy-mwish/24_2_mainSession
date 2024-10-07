import random
import numpy as np

# reward 수정하기 

NUM_OF_PLAYOUT = 10

class Agent:
    def __init__(self, env):
        self.env = env
    
    def _playout(self, state):
        if state.is_lose():
            return self.env.reward['lose']
        
        if state.is_draw():
            return  self.env.reward['draw']
        
        # 다음 상태의 상태 평가
        return -self._playout(state.next(self.random_available_action(state)))
    
    def random_available_action(self, state):
        available_actions = state.available_actions
        return random.choice(available_actions)

    def mc_action(self, state):
        available_actions = state.available_actions
        val_per_action = np.zeros(shape=len(available_actions))

        for i, action in enumerate(available_actions):
            for _ in range(NUM_OF_PLAYOUT):
                val_per_action[i] += - self._playout(state.next(action))
            val_per_action[i] /= NUM_OF_PLAYOUT

        print(f"available_actions : {available_actions}")
        print(f"val_per_action : {val_per_action}")
        return available_actions[self._argmax(val_per_action)]
    
    def _argmax(self, collection):
        return np.arange(len(collection))[collection == np.max(collection)][0] # lst -> int