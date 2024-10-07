import random
import numpy as np
from node import *

# reward 수정하기 

class Agent:
    def __init__(self, env):
        self.env = env

    def random_action(self):
        '''
        totaly random action.
        '''
        action = random.choice(self.env.action_space)
        return action
    
    def random_available_action(self, state):
        available_actions = state.available_actions
        return random.choice(available_actions)
    
    def _argmax(self, collection):
        return np.arange(len(collection))[collection == np.max(collection)][0] # lst -> int

    def mcts_action(self, state):
        root_node = Node(state)
        root_node.expand()

        for _ in range(100):
            root_node.evaluate()

        available_actions = state.available_actions
        n_lst = []

        for child_node in root_node.child_nodes:
            n_lst.append(child_node.n)
        
        return available_actions[self._argmax(n_lst)]


    