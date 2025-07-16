import numpy as np
import sys
import math

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.col_count = 7
        self.action_size = self.col_count
        self.in_a_row = 4

    def get_initial_state(self): # returns empty board
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(self, state, action, player):
        '''
        state = current board
        action = column index (0-6)
        player = current player (1 or -1)

        find lowest empty row in the column, put the player in that position, return board and the row it is on
        '''
        state = state.copy()
        empty_rows = np.where(state[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return state, None  # or return None
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state, row

    def get_valid_moves(self, state):
        # return all valid columns given a board
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, row, col):
        player = state[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diag1, diag2
        for dr, dc in directions:
            count = 1

            # Check in both directions from the last placed piece
            for step in [-1, 1]:
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.row_count and 0 <= c < self.col_count and state[r][c] == player:
                    count += 1
                    if count >= self.in_a_row:
                        return True
                    r += dr * step
                    c += dc * step

        return False

    def get_value_and_terminated(self, state, row, action):
        if self.check_win(state, row, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return -0.05, True
        return 0, False

    def get_opponent(self, player):
        # switches player
        return -player

    def get_opponent_value(self, value):
        # flip result for the other player's perspective
        return -value

    def change_perspective(self, state, player):
        # flip the board
        return state * player
    
    def print_board(self, state):
        disp = {0: '.', 1: 'Y', -1: 'R'}
        for row in state:
            print(' '.join(disp[int(cell)] for cell in row))
        print('1 2 3 4 5 6 7\n')


class MCTSNode:
    __slots__ = ('game','state','player','parent','action', 'children','N','W','untried')

    def __init__(self, game, state, player, parent=None, action=None):
        self.game    = game
        self.state   = state            # full 6×7 board
        self.player  = player           # who JUST moved to get here
        self.parent  = parent
        self.action  = action           # the column that was played
        self.children= []               
        self.N       = 0                # visit count
        self.W       = 0.0              # total value
        # moves we haven’t expanded yet
        self.untried = list(np.where(game.get_valid_moves(state)==1)[0])

    def is_fully_expanded(self):
        return not self.untried

    def select_ucb(self, c):
        # pick child with max UCB score
        return max(
            self.children,
            key=lambda n: (n.W/n.N) + c * math.sqrt(math.log(self.N) / n.N)
        )

    def expand(self):
        # pop one untried move
        a = self.untried.pop()
        next_state, row = self.game.get_next_state(self.state, a, -self.player)
        next_player= self.game.get_opponent(self.player)
        child = MCTSNode(self.game, next_state, next_player, self, a)
        self.children.append(child)
        return child

    def rollout(self):
        s, p = self.state.copy(), self.player
        while True:
            moves = np.where(self.game.get_valid_moves(s)==1)[0]
            if moves.size == 0:
                return 0
            a = np.random.choice(moves)
            s, r = self.game.get_next_state(s, a, -p)
            val, done = self.game.get_value_and_terminated(s, r, a)
            if done:
                return val
            p = self.game.get_opponent(p)

    def backpropagate(self, value, root_player):
        node = self
        while node:
            node.N += 1
            # if this node was played by root_player, add value; else subtract
            node.W += value if node.player == root_player else -value
            node = node.parent


class UCT:
    def __init__(self, game, gameargs):
        self.game = game
        self.num_simulations = gameargs['num_searches']
        self.c = math.sqrt(2)

    def search(self, root_state, root_player):
        # root_player is about to move on root_state
        root = MCTSNode(self.game, root_state, root_player)

        for _ in range(self.num_simulations):
            node = root
            # 1) SELECTION
            while node.is_fully_expanded() and node.children:
                node = node.select_ucb(self.c)
            # 2) EXPANSION
            if node.untried:
                node = node.expand()
            # 3) SIMULATION
            value = node.rollout()
            # 4) BACKPROPAGATION
            node.backpropagate(value, root_player)

        # pick the child with highest visit count
        best_child = max(root.children, key=lambda n: n.N)
        return best_child.action


class QLearningAgent:
    def __init__(self, feature_fn, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, eps_decay=0.9995, eps_min=0.1):
        self.feature_fn = feature_fn
        self.action_size = action_size
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.eps_decay, self.eps_min = epsilon, eps_decay, eps_min
        self.Q = {}   # dict mapping ((f1,f2,f3,f4), action) -> value

    def get_q(self, state_feats, a):
        return self.Q.get((state_feats, a), 0.0)

    def choose_action(self, board, valid_moves, player):
        feats = self.feature_fn(board, player)
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.where(valid_moves==1)[0])
        # greedy
        qs = [self.get_q(feats, a) if valid_moves[a] else -np.inf
              for a in range(self.action_size)]
        return int(np.argmax(qs))

    def update(self, board, action, reward, next_board, next_valid, done, player):
        s  = self.feature_fn(board, player)
        s2 = self.feature_fn(next_board, player)
        # max_a' Q(s',a')
        future = 0 if done else max(self.get_q(s2,a) for a in range(self.action_size))
        old = self.get_q(s, action)
        self.Q[(s, action)] = old + self.alpha * (reward + self.gamma*future - old)
        # decay ε
        if done:
            self.epsilon = max(self.epsilon*self.eps_decay, self.eps_min)





def count_runs(board: np.ndarray, player: int, length: int) -> int:
    rows, cols = board.shape
    opp = -player
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    total = 0

    for dr, dc in directions:
        # for each possible starting cell
        for i in range(rows):
            for j in range(cols):
                end_i = i + (length-1)*dr
                end_j = j + (length-1)*dc
                # must fit the run in‑board
                if not (0 <= end_i < rows and 0 <= end_j < cols):
                    continue

                # check the run itself
                run_ok = True
                for k in range(length):
                    if board[i + k*dr, j + k*dc] != player:
                        run_ok = False
                        break
                if not run_ok:
                    continue

                # check “pre” cell isn’t blocked by opponent
                pi, pj = i - dr, j - dc
                if 0 <= pi < rows and 0 <= pj < cols and board[pi, pj] == opp:
                    continue

                # check “post” cell isn’t blocked by opponent
                qi, qj = i + length*dr, j + length*dc
                if 0 <= qi < rows and 0 <= qj < cols and board[qi, qj] == opp:
                    continue

                total += 1

    return total

def extract_features(board, player):
    return (
      np.sum(board[:, board.shape[1]//2] == player),  # center_count
      count_runs(board, player, 2),
      count_runs(board, player, 3),
      count_runs(board, -player, 3),
    )

def train_q_agent(episodes, game, agent):
    for ep in range(episodes):
        state = game.get_initial_state().copy()
        player = 1
        done = False
        last = []   # to store (board, action, player) for updates
        while not done:
            valid = game.get_valid_moves(state)
            act = agent.choose_action(state, valid, player)
            next_s = state.copy()
            next_s, row = game.get_next_state(next_s, act, player)
            val, done = game.get_value_and_terminated(next_s, row, act)
            # reward from current player’s POV
            reward = 0 if not done else (1 if val==1 else -1 if val== -1 else 0)
            # update for the agent who just moved
            agent.update(state, act, reward, next_s, game.get_valid_moves(next_s), done, player)
            state, player = next_s, game.get_opponent(player)
        # optional: print progress every 1000 eps
        if ep % 1000 == 0:
            print(f"Episode {ep}, (epsilon)={agent.epsilon:.3f}")

def play_game_vs_uct(ql_agent, uct_agent, game):
    state  = game.get_initial_state().copy()
    # 1 = QL’s turn, -1 = UCT’s turn
    player = 1

    print(" ----- Initial state of board ----- ")
    game.print_board(state)

    # Start game, constantly swapping
    while True:
        # Get all the valid moves on the board
        valid = game.get_valid_moves(state)

        if player == 1: # qlearning choose move
            move = int(input("Input move")) - 1
            #move = ql_agent.choose_action(state, valid, player)
            label = "QL"
        else:           #  uct algorithm choose move
            move = uct_agent.search(state, player)
            label = "UCT"

        state, row = game.get_next_state(state.copy(), move, player)
        print(f"{label} plays column {move+1}")

        # print the move played on the board
        game.print_board(state)

        # check if the move won the game by the player
        done = game.check_win(state, row, move)

        if done:
            print(f"Game over, {label} wins")
            break

        # switch player
        player *= -1




def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']

    algorithm = lines[0]
    player_char = lines[1]
    board_lines = lines[2:]
    board = np.zeros((6, 7))

    for i in range(6):
        for j in range(7):
            char = board_lines[i][j]
            if char == 'R':
                board[i][j] = -1
            elif char == 'Y':
                board[i][j] = 1

    return algorithm, player_char, board


def main():
    import time
    start_time = time.time()

    game = ConnectFour()

    num_simulations = 10000
    mode = 'none'

    game_args = {
        'num_searches': num_simulations,  # simulation count
        'mode': mode             # verbose or brief or none
    }

    # train qlearning agent
    ql_agent = QLearningAgent(extract_features, game.action_size)
    #train_q_agent(3000, game, ql_agent)
    ql_agent.epsilon = 0

    uct = UCT(game, game_args)

    print("\n--- Playing one game: Q-learning vs UCT ---")
    play_game_vs_uct(ql_agent, uct, game)

    # Timer end
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Function executed in {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    main()
