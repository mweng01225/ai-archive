import random
import numpy as np
from collections import defaultdict
from connect7 import ConnectFour

# board stuff
ROWS, COLS = 6, 7

class FeatureQAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, eps_decay=0.9999, eps_min=0.01):
        # Q-table mapping (feature_tuple, action) -> value
        self.Q = defaultdict(lambda:1.0)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

    def extract_features(self, board, player):
        """
        some hand crafted features 
        f1 = open-three for player
        f2 = open-three for opponent
        f3 = open-two for player
        f4 = open-two for opponent
        f5 = center-column control
        """
        def count_open_n(b, pl, n):
            # counts windows of length 4 with exactly n pieces of player and the rest empty
            count = 0
            # horizontal
            for r in range(ROWS):
                for c in range(COLS - 3):
                    window = b[r, c:c+4]
                    if np.count_nonzero(window == pl) == n and np.count_nonzero(window == 0) == 4 - n:
                        count += 1
            # vertical
            for c in range(COLS):
                for r in range(ROWS - 3):
                    window = b[r:r+4, c]
                    if np.count_nonzero(window == pl) == n and np.count_nonzero(window == 0) == 4 - n:
                        count += 1
            # diagonal (\)
            for r in range(ROWS - 3):
                for c in range(COLS - 3):
                    window = np.array([b[r + i, c + i] for i in range(4)])
                    if np.count_nonzero(window == pl) == n and np.count_nonzero(window == 0) == 4 - n:
                        count += 1
            # diagonal (/)
            for r in range(3, ROWS):
                for c in range(COLS - 3):
                    window = np.array([b[r - i, c + i] for i in range(4)])
                    if np.count_nonzero(window == pl) == n and np.count_nonzero(window == 0) == 4 - n:
                        count += 1
            return count

        f1 = count_open_n(board, player, 3)
        f2 = count_open_n(board, -player, 3)
        f3 = count_open_n(board, player, 2)
        f4 = count_open_n(board, -player, 2)
        # diff between player and opponent count
        center = board[:, COLS // 2]
        f5 = int(np.count_nonzero(center == player) - np.count_nonzero(center == -player))
        return (f1, f2, f3, f4, f5)

    def get_state_key(self, board, player):
        return self.extract_features(board, player)

    def choose_action(self, board, valid_actions, player):
        # Immediate win |i dont think this works
        for a in valid_actions:
            if self._can_win_in_one(board, player, a):
                return a

        # 2) Immediate block
        opp = -player
        for a in valid_actions:
            # simulate you playing a
            temp = ConnectFour()
            temp.board = board.copy()
            temp.turn  = player
            if not temp.get_next_state(a):
                continue

            # now check if opponent can win from that resulting board
            next_moves = temp.get_valid_moves(temp.board)
            if any(self._can_win_in_one(temp.board, opp, b) for b in next_moves):
                return a

        # epsilon-greedy + argmax-Q
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = [(self.Q[(self.get_state_key(board, player), a)], a) for a in valid_actions]
        max_q = max(q for q,_ in q_vals)
        best = [a for q,a in q_vals if q==max_q]
        return random.choice(best)

    def _can_win_in_one(self, board, player, col):
        """Return True if playing `col` as `player` immediately wins."""
        temp = ConnectFour()
        temp.board = board.copy()
        temp.turn  = player
        valid = temp.get_next_state(col)
        if not valid:
            return False
        done, winner = temp.check_win()
        return done and winner == player

    def update(self, board, action, reward, next_board, next_valid, done, player):
        state = self.get_state_key(board, player)
        sa = (state, action)
        q_sa = self.Q[sa]
        if done:
            target = reward
        else:
            next_state = self.get_state_key(next_board, -player)
            # look ahread at next best action
            future = max(self.Q[(next_state, a)] for a in next_valid)
            target = reward + self.gamma * future

        # update rule
        self.Q[sa] += self.alpha * (target - q_sa)

    def decay_epsilon(self):
        # decays epsilon value
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def train_self_play(self, env, episodes=10000):
        for ep in range(episodes):
            board = env.reset()
            player = 1
            done = False
            while not done:
                # get legal moves
                valid_cols = env.get_valid_moves(board)

                #choose and apply action
                action = self.choose_action(board, valid_cols, player)
                next_board, reward, done = env.step(action)

                # get next valid moves
                next_valid_cols = env.get_valid_moves(next_board)

                # q update
                self.update(board, action, reward, next_board, next_valid_cols, done, player)

                #switch player and state
                board = next_board
                player *= -1
                
            # decay and print
            self.decay_epsilon()
            if ep % 100 == 0:
                print(f"Episode = {ep}")