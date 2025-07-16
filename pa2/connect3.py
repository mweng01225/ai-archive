import numpy as np
import sys
import math

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.col_count = 7
        self.action_size = self.col_count
        self.in_a_row = 4

    def get_initial_state(self): # returns empty board for rollouts
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(self, state, action, player):
        empty_rows = np.where(state[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return state  # or return None
        # find lowest empty row in the column, put the player in that position, return board
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        # get valid moves
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        # returns true of action leads to win
        if action is None:
            return False

        row = np.max(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if r < 0 or r >= self.row_count or c < 0 or c >= self.col_count or state[r][c] != player:
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 or
            (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 or
            (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 or
            (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
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


class Node: # for mcts
    def __init__(self, game, args, parent=None, action_taken=None):
        self.game = game                 # connectFour object
        self.args = args                 # {num_searches, mode}
        self.parent = parent             #parent node of this node
        self.action_taken = action_taken # move that led to this node
        self.children = []               # list of child nodes
        self.visit_count = 0
        self.value_sum = 0
        self.expandable_moves = None

    def is_fully_expanded(self):
        # returns true if all valid moves from this node have been expanded
        return self.expandable_moves is not None and np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def get_state(self):
        # reconstruct board state by replaying the moves from the root to this node
        moves = []
        node = self
        while node.parent is not None:
            moves.append(node.action_taken)
            node = node.parent
        state = self.game.get_initial_state()
        player = 1
        for move in reversed(moves):
            state = self.game.get_next_state(state, move, player)
            player = self.game.get_opponent(player)
        return state

    def select(self): #select random children based on instruction
        return np.random.choice(self.children)

    def expand(self):
        if self.expandable_moves is None:
            self.expandable_moves = self.game.get_valid_moves(self.get_state())
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0

        child = Node(self.game, self.args, self, action)
        self.children.append(child)

        if self.args['mode'] == 'verbose':
            print(f"NODE ADDED (Move: {action + 1})")

        return child

    def simulate(self): #rollout policy
        # reconstruct board state
        state = self.get_state()
        value, is_terminal = self.game.get_value_and_terminated(state, self.action_taken)
        value = self.game.get_opponent_value(value) # adjust value for opponent


        rollout_state = state.copy()
        rollout_player = 1 # start rollout with current player
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0]) # select a random move
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player) 
            if self.args['mode'] == 'verbose':
                print(f"Move selected: {action + 1}") # print rollout step


            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value) # flip result
                if self.args['mode'] == 'verbose':
                    print(f"TERMINAL NODE VALUE: {value}")
                return value
            
            rollout_player = self.game.get_opponent(rollout_player)# switch player

    def backpropagate(self, value):
        self.value_sum += value # update value for this node
        self.visit_count += 1   # visit count + 1

        if self.args['mode'] == 'verbose':
            print(f"Updated values (Move: {'' if self.action_taken is None else self.action_taken + 1}): wi: {self.value_sum}, ni: {self.visit_count}")

        value = self.game.get_opponent_value(value) #flip value for parent node

        # update parents
        if self.parent is not None:
            self.parent.backpropagate(value)


class UCT:
    def __init__(self, game, args):
        self.game = game
        self.args = args # {num_seraches, mode}

    def search(self, state):
        root = Node(self.game, self.args)

        for _ in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = self.select_ucb(node)  # using UCB to select best child

            value, is_terminal = self.game.get_value_and_terminated(node.get_state(), node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_values = [None] * self.game.action_size
        best_action = None
        best_value = -float("inf")

        # compute values from expanded children
        for child in root.children:
            wi = child.value_sum
            ni = child.visit_count
            if ni > 0:
                v = wi / ni
                action_values[child.action_taken] = round(v, 2)
                if v > best_value:
                    best_value = v
                    best_action = child.action_taken
            else:
                action_values[child.action_taken] = "Null"

        # columns are marked as null if not expanded
        valid_moves = self.game.get_valid_moves(state)
        for i in range(self.game.action_size):
            if valid_moves[i] == 0:
                action_values[i] = "Null"

        if self.args['mode'] != 'none':
            for i, val in enumerate(action_values):
                print(f"Column {i + 1}: {val}")

        return best_action

    def select_ucb(self, node):
        c = self.args.get('exploration_constant', math.sqrt(2)) #exploration constant either in args or not provided, default to sqrt 2

        best_score = -float("inf")
        best_child = None
        for child in node.children:
            if child.visit_count == 0:
                ucb = float("inf")  # Always explore unvisited nodes
            else:
                q = child.value_sum / child.visit_count
                ucb = q + c * math.sqrt(math.log(node.visit_count) / child.visit_count)


            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child




class QLearningAgent:
    def __init__(self, feature_fn, action_size,
                 alpha=0.1, gamma=0.95, epsilon=1.0, eps_decay=0.9995, eps_min=0.1):
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
    """
    Count the number of runs of exactly `length` for `player` on the board
    that are not blocked by the opponent at either end.

    Args:
      board: 2D np.array of shape (6,7) with values {0, 1, -1}.
      player: 1 or -1.
      length: run‐length to look for (e.g. 2, 3).

    Returns:
      Integer count of unblocked runs.
    """
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
            next_s = game.get_next_state(next_s, act, player)
            val, done = game.get_value_and_terminated(next_s, act)
            # reward from current player’s POV
            reward = 0 if not done else (1 if val==1 else -1 if val== -1 else 0)
            # update for the agent who just moved
            agent.update(state, act, reward, next_s,
                         game.get_valid_moves(next_s), done, player)
            state, player = next_s, game.get_opponent(player)
        # optional: print progress every 1000 eps
        if ep % 1000 == 0:
            print(f"Episode {ep}, ε={agent.epsilon:.3f}")

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
            move = ql_agent.choose_action(state, valid, player)
            label = "QL"
        else:           #  uct algorithm choose move
            s2 = game.change_perspective(state.copy(), player)
            move = uct_agent.search(s2)
            label = "UCT"

        state = game.get_next_state(state.copy(), move, player)
        print(f"{label} plays column {move+1}")

        # print the move played on the board
        game.print_board(state)

        # check if the move won the game by the player
        done = check_win(state, move, player)

        if done:
            print(f"Game over, {label} wins")
            break

        # switch player
        player *= -1

def check_win(board, col, piece):
    ROWS, COLS = 6, 7
    
    # Find the row of the most recent move
    row = -1
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == piece:
            row = r
            break
    if row == -1:
        return False  # Piece not found in column
    
    # Directions to check: (delta_row, delta_col)
    directions = [
        (0, 1),   # Horizontal
        (1, 0),   # Vertical
        (1, 1),   # Diagonal down-right
        (1, -1),  # Diagonal down-left
    ]
    
    for dr, dc in directions:
        count = 1
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == piece:
            count += 1
            r += dr
            c += dc
            
        # Check in the negative direction
        r, c = row - dr, col - dc
        while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == piece:
            count += 1
            r -= dr
            c -= dc
        
        if count >= 4:
            return True
    
    return False


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
    train_q_agent(100, game, ql_agent)
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
