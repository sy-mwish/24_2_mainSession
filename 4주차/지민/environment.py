import pandas as pd
from state import *
from IPython.display import display

class TicTacToeEnvironment:
    def __init__(self):
        self.state = State()

        self.board_size = (3,3)
        self.action_space = range(self.board_size[0]*self.board_size[1])
        self.n_actions = len(self.action_space)

        self.reward = {'win' : 10, 'lose' : -10, 'draw' : 0, 'progress' : 0}

    def step(self, action):
        my_actions = self.state.my_actions.copy()
        enemy_actions = self.state.enemy_actions.copy()

        my_actions.append(action)
        
        next_state = State(my_actions, enemy_actions) # action으로 인해 발생한 나의 next state
        self.state = State(self.state.enemy_actions, my_actions) # 다음 스텝을 위해 편의상 상대-나를 뒤집은 state

        if next_state.is_win():
            reward, done = self.reward['win'], True

        elif next_state.is_draw():
            reward, done = self.reward['draw'], True

        elif next_state.is_lose():
            reward, done = self.reward['lose'], True
            
        else:
            reward, done = self.reward['progress'], False

        return self.state, next_state, reward, done

    def reset(self):
        self.state = State()
        return self.state

    def render(self, state):

        board = state.board[0] + (-1 * state.board[1]) if state.is_first_player() else state.board[1] + (-1 * state.board[0])

        def pattern(x):
            if x == 1:
                return 'O'
            elif x == 0:
                return ' '
            else:
                return 'X'
            
        df = pd.DataFrame(board)
        df = df.map(pattern)
        display(df)