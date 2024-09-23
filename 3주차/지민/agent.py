import random
import numpy as np

# reward 수정하기 

class Agent:
    def __init__(self, env):
        self.env = env

    def _get_available_actions(self, state):
        indices = np.arange(self.env.n_actions)
        available_actions = indices[(state == 0).reshape(-1)].tolist()
        return available_actions

    def random_action(self):
        '''
        totaly random action.
        '''
        action = random.choice(self.env.action_space)
        return action
    
    def random_available_action(self, state):
        available_actions = state.available_actions
        return random.choice(available_actions)

    # 미니맥스법을 활용한 상태 가치 계산
    def mini_max(self, state):
        # 패배 시, 상태 가치 -10
        if state.is_lose():
            return self.env.reward['lose']
        
        # 무승부 시, 상태 가치 0
        if state.is_draw():
            return  self.env.reward['draw']

        # if state.is_win(): # 이 코드에서 step은 상대편의 state라서 승리는 판정되지 않음 
        #     print('win')
        #     return 10 

        max_value = float("-inf")

        for action in state.available_actions:
            value = -self.mini_max(state.next(action))
            max_value = max(max_value, value)

        return max_value

    def mini_max_action(self, state):
        best_action = None
        best_value = float("-inf")

        action_values = []

        for action in state.available_actions:
            value = - self.mini_max(state.next(action))
            action_values.append(value)

            if value > best_value:
                best_action = action
                best_value = value

        print("Available actions:", state.available_actions)        
        print("Action values:", action_values)

        if action_values.count(max(action_values)) == 1:
            best_action = best_action

        else:
            max_indices = list(filter(lambda i: action_values[i] == max(action_values), range(len(action_values))))
            best_action = state.available_actions[random.choice(max_indices)] 

        return best_action
    
    # 미니맥스법을 활용한 상태 가치 계산 + depth를 포함 
    def mini_max_with_depth(self, state, depth=0):
        # 패배 시, 상태 가치 -10
        if state.is_lose():
            return self.env.reward['lose'] + depth
        
        # 무승부 시, 상태 가치 0
        if state.is_draw():
            return  self.env.reward['draw']

        max_value = float("-inf")

        for action in state.available_actions:
            value = - self.mini_max_with_depth(state.next(action), depth+1)
            max_value = max(max_value, value)

        return max_value

    def mini_max_action_with_depth(self, state):
        best_action = None
        best_value = float("-inf")

        action_values = []

        for action in state.available_actions:
            value = - self.mini_max_with_depth(state.next(action))
            action_values.append(value)

            if value > best_value:
                best_action = action
                best_value = value

        print("Available actions:", state.available_actions)        
        print("Action values:", action_values)

        return best_action
    
    def alpha_beta(self, state, alpha, beta):
        # 패배 시, 상태 가치 -10
        if state.is_lose():
            return self.env.reward['lose']
        
        # 무승부 시, 상태 가치 0
        if state.is_draw():
            return self.env.reward['draw']

        # 합법적인 수의 상태 가치 계산
        for action in state.available_actions:
            # 상대방의 턴에서 탐색하므로, 상태 가치를 -로 반전
            score = -self.alpha_beta(state.next(action), -beta, -alpha)

            # 현재 노드에서 알파 값을 업데이트
            if score > alpha:
                alpha = score

            # 가지치기 발생
            if alpha >= beta:
                return alpha

        # 탐색된 수 중 최대값 반환
        return alpha

    def alpha_beta_action(self, state):
        best_action = None
        alpha = -float('inf')
        
        action_values = []

        for action in state.available_actions:
            score = -self.alpha_beta(state.next(action), -float('inf'), -alpha)
            action_values.append(score)

            if score > alpha:
                best_action = action
                alpha = score
        
        print("Available actions:", state.available_actions)        
        print("Action values:", action_values)

        # 만약 여러 개의 최고값이 존재한다면 그 중 하나를 선택
        max_value = max(action_values)
        best_actions = [action for action, value in zip(state.available_actions, action_values) if value == max_value]
        
        # 여러 개의 동일한 최고값이 있을 때 랜덤으로 선택 
        if len(best_actions) > 1:
            best_action = random.choice(best_actions)

        return best_action
