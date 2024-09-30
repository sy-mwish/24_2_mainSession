import numpy as np 

# -------
# magic num handling
# -------


class State:
    def __init__(self, my_actions=None, enemy_actions=None):
        self.my_actions = [] if my_actions is None else my_actions
        self.enemy_actions = [] if enemy_actions is None else enemy_actions
        self.board = self.create_board(self.my_actions, self.enemy_actions)
        self.available_actions = self._get_available_actions()

    def next(self, action):
        my_actions = self.my_actions.copy()
        my_actions.append(action)
        return State(self.enemy_actions, my_actions)
    
    def create_board(self, my_actions, enemy_actions):
        # 전체 state 
        total_board = np.zeros(shape=(2,3,3))

        # 내 말과 상대방 말이 놓인 보드를 원핫인코딩으로 표현 
        my_board, enemy_board = np.zeros(9), np.zeros(9)

        my_board[my_actions] = 1
        enemy_board[enemy_actions] = 1

        total_board[0] = my_board.reshape((3,3))
        total_board[1] = enemy_board.reshape((3,3))

        return total_board 

    def _get_available_actions(self):
        my_actions_set = set(self.my_actions)
        enemy_actions_set = set(self.enemy_actions)

        return list(set(range(9)) - my_actions_set - enemy_actions_set)

    def is_win(self):
        my_state = self.board[0]

        rows_win = (my_state == 1).sum(axis=0).max() == 3
        cols_win = (my_state == 1).sum(axis=1).max() == 3
        diag_win = np.diag(my_state).sum() == 3
        anti_diag_win = np.diag(np.fliplr(my_state)).sum() == 3

        return rows_win or cols_win or diag_win or anti_diag_win

    def is_draw(self):
        return (np.sum(self.board[0]) + np.sum(self.board[1])) >= 9

    def is_lose(self):
        enemy_state = self.board[1]

        rows_lose = (enemy_state == 1).sum(axis=0).max() == 3
        cols_lose = (enemy_state == 1).sum(axis=1).max() == 3
        diag_lose = np.diag(enemy_state).sum() == 3
        anti_diag_lose = np.diag(np.fliplr(enemy_state)).sum() == 3

        return rows_lose or cols_lose or diag_lose or anti_diag_lose

    def is_done(self):
        return self.is_win() or self.is_draw() or self.is_lose()
    
    def is_first_player(self):
        return len(self.my_actions) == len(self.enemy_actions)