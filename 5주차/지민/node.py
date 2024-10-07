import random
import math
import numpy as np

class Node:
    def __init__(self, state):
        self.state = state
        self.v = 0
        self.n = 0
        self.child_nodes = []

    def random_available_action(self, state):
        available_actions = state.available_actions
        return random.choice(available_actions)
    
    def _playout(self, state):
        if state.is_lose():
            return -1 
        
        if state.is_draw():
            return  0
        
        # 다음 상태의 상태 평가
        return -self._playout(state.next(self.random_available_action(state)))
    
    def _argmax(self, collection):
        return np.arange(len(collection))[collection == np.max(collection)][0] # lst -> int

    def evaluate(self):
        # 게임 종료 시 
        if self.state.is_done():
            value = -1 if self.state.is_lose() else 0   # magic num

            self.v += value
            self.n += 1

            return value
        
        # 자식 노드가 존재하지 않는 경우
        # 없는게 아니라 아직 확장되지 않은 것이다. 
        if len(self.child_nodes) == 0:
            value = self._playout(self.state)

            self.v += value
            self.n += 1

            if self.n == 10: # 왜 10인지 찾기 
                self.expand()

            return value
        
        # 종결되지 않았지만, 자식 노드가 존재하는 경우
        else:
            child_node = self.get_next_child_node()
            value = -child_node.evaluate() 

            self.v += value
            self.n += 1

            return value
        
    def expand(self):
        available_actions = self.state.available_actions 
        self.child_nodes = []
        for action in available_actions:
            self.child_nodes.append(Node(self.state.next(action)))

    def get_next_child_node(self):
        for child_node in self.child_nodes:
            # 우선 탐색 
            if child_node.n == 0:
                return child_node

        # UCB1
        t = 0
        for child_node in self.child_nodes:
            t += child_node.n

        ucb1_values = []

        for child_node in self.child_nodes:
            ucb1_values.append(-child_node.v/child_node.n+(2*math.log(t)/child_node.n)**0.5)

        return self.child_nodes[self._argmax(ucb1_values)]

