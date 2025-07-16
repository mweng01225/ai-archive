import random
import sys
import numpy as np
import time

class ConnectFour:
    def __init__(self):
        self.turn = 1 # Y goes first
        self.board = np.zeros(shape=(6, 7))


    def print_board(self):
        disp = {0: '.', 1: 'Y', -1: 'R'}
        for row in reversed(self.board):
            print(' '.join(disp[int(cell)] for cell in row))
        print('1 2 3 4 5 6 7\n')
        if self.turn == -1:
            print("UCT's turn, Please wait...")
            

    def get_next_state(self, move = None, print_move = False):
        if not move:
            # Move not provided, take user input
            try:
                print("Your turn, enter a number between 1 and 7: ", end="")
                move = int(input())
            except ValueError:
                return False
        if print_move:
            print(f"Column chosen = {move}")
        # Move is provided, make sure the input is 1 indexed
        if move not in [1, 2, 3, 4, 5, 6, 7]:
            print("Not a valid input")
            return False
        
        # put into board zero indexed
        for i in range(6):
            if self.board[i, move - 1] == 0:
                self.board[i, move - 1] = self.turn
                self.switch_turn()
                return True
        return False


    def check_win(self):
        winner = ConnectFour.check_rows(self.board)
        if winner is not None:
            return (True, winner)

        winner = ConnectFour.check_cols(self.board)
        if winner is not None:
            return (True, winner)

        winner = ConnectFour.check_diag(self.board)
        if winner is not None:
            return (True, winner)

        if ConnectFour.check_tie(self.board):
            return (True, None)

        return (False, None)

    @staticmethod
    def check_rows(board):
        for y in range(6):
            row = list(board[y, :])
            for x in range(4):
                if row[x : x + 4].count(row[x]) == 4:
                    if row[x] != 0:
                        return row[x]
        return None

    @staticmethod
    def check_cols(board):

        for x in range(7):
            col = list(board[:, x])
            for y in range(3):
                if col[y : y + 4].count(col[y]) == 4:
                    if col[y] != 0:
                        return col[y]
        return None

    @staticmethod
    def check_diag(board):
        rows, cols = board.shape
        for r in range(rows - 3):
            for c in range(cols - 3):
                w = board[r:r+4, c:c+4].diagonal()
                if abs(w.sum()) == 4 and (w[0] != 0):
                    return w[0]
        for r in range(rows - 3):
            for c in range(3, cols):
                w = np.fliplr(board[r:r+4, c-3:c+1]).diagonal()
                if abs(w.sum()) == 4 and (w[0] != 0):
                    return w[0]
        return None

    @staticmethod
    def check_tie(board):
        return bool(np.all(board != 0))

    def switch_turn(self):
        self.turn *= -1

    def get_valid_moves(self, state):
        return [c + 1 for c in range(7) if state[-1, c] == 0]


class UCT:
    def __init__(self, simulations):
        self.simulations = simulations

    def search(self, node):
        for _ in range(self.simulations):
            # select and expand
            leaf = self.select(node)
            if leaf is None:
                return (-1, -1)
            
            # simulation
            simulation_result = self.rollout(leaf)

            # backpropagation
            self.backpropagate(leaf, simulation_result)
        # from next best state get move coordinates


        # select best child within children
        max_visit = 0
        selected = None
        for child in node.children:
            if child.visit_count > max_visit:
                max_visit = child.visit_count
                selected = child


        if selected is None:
            return (-1, -1)
        
        for j in range(6):
            for i in range(7):
                if selected.board[j][i] != node.board[j][i]:
                    return (j, i)
        return (-1, -1)

    def select(self, node):
        # select best child if node is fully expanded
        while self.expanded(node):
            child = self.select_ucb(node)
            # break if select_uct returns the same node back
            if child == node:
                break
            node = child


        # return terminal node
        if node.terminal:
            return node
        
        # expand node and return it for rollout
        node.add_child()

        if node.children:
            return next((child for child in node.children if child.visit_count == 0), node)
        return node

    def select_ucb(self, node):
        c = np.sqrt(2)
        best_uct = -np.inf
        best_node = None
        for child in node.children:
            if child.visit_count == 0:
                return child # this should not happen | just to "avoid" zero Division Error

            q = (child.value_sum / child.visit_count)
            uct = q + c * np.sqrt(np.log(node.visit_count) / child.visit_count)
            if uct > best_uct:
                best_uct = uct
                best_node = child

        # Avoid error if node has no children
        if best_node is None:
            return node
        return best_node

    def expanded(self, node):    
        # check if a node is fully expanded
        legal_moves = np.count_nonzero(node.board[5] == 0)
        # if num children < number of possible moves, its not expanded
        if len(node.children) < legal_moves:
            return False
        # check if all children have been visited
        visits = np.array([child.visit_count for child in node.children])
        return np.all(visits > 0)

    def rollout(self, node):
        board = node.board
        turn = node.turn

        if node.terminal:
            return self.result(board)
        
        while True:
            # switch turn
            turn *= -1
            # get moves from current board
            moves = self.get_valid_moves(board, turn)

            if not moves: # no moves left
                return self.result(board)

            # select next board randomly
            board = random.choice(moves)
            # check if state is terminal
            terminal = self.result(board)
            if terminal != 0:
                return terminal

    def get_valid_moves(self, board, turn):
        moves = []
        for i in range(7):
            if board[5, i] == 0:
                for j in range(6):
                    if board[j, i] == 0:
                        tmp = board.copy()
                        if turn == -1:
                            tmp[j, i] = 1
                        else:
                            tmp[j, i] = -1
                        moves.append(tmp)
                        break
        return moves

    def result(self, board):
        winner = ConnectFour.check_rows(board)
        if winner is not None:
            return winner

        winner = ConnectFour.check_cols(board)
        if winner is not None:
            return winner

        winner = ConnectFour.check_diag(board)
        if winner is not None:
            return winner

        return None

    def backpropagate(self, node, winner):
        # if win, increment
        if node.turn == winner:
            node.value_sum += 1
        # increase visit number
        node.visit_count += 1
        # base case
        if node.parent is None:
            return
        # backprop
        self.backpropagate(node.parent, winner)



class Node:
    def __init__(self, parent, board, turn):
        self.value_sum = 0  # sum of rollout outcomes
        self.visit_count = 0  # number of visits
        self.parent = parent
        self.board = board
        # flip turns when creating new nodes
        self.turn = turn * -1
        self.children = []
        self.terminal = self.check_terminal()
        self.expanded = False

    def check_terminal(self):
        if ConnectFour.check_rows(self.board):
            return True

        if ConnectFour.check_cols(self.board):
            return True

        if ConnectFour.check_diag(self.board):
            return True

        if ConnectFour.check_tie(self.board):
            return True

        return False

    def add_child(self):
        # node already expanded
        # If we've already generated *all* children, do nothing
        if self.expanded:
            return

        # On first call, build a list of all valid (row, col) moves
        if not hasattr(self, "_untried_moves"):
            n_rows, n_cols = self.board.shape
            self._untried_moves = []
            for col in range(n_cols):
                # only consider if column not full
                if self.board[n_rows-1, col] == 0:
                    # find the first empty row (from top idx=0 down)
                    empties = np.where(self.board[:, col] == 0)[0]
                    if empties.size > 0:
                        row = empties[0]
                        self._untried_moves.append((row, col))

        # If no moves left, mark expanded and return
        if not self._untried_moves:
            self.expanded = True
            return

        # Pop one untried move and create its child
        row, col = self._untried_moves.pop()
        new_board = self.board.copy()
        # place the new piece; original logic used tmp[j,i] = -self.turn
        new_board[row, col] = -self.turn
        self.children.append(Node(self, new_board, self.turn))



class QLearningAgent:
    def __init__(self,
        alpha = 0.05,            # learning rate
        gamma = 0.95,            # discount
        epsilon = 1,             # full random
        epsilon_decay = 0.99995,  # decay into greedy
        eps_min = 0.01       # final minimum randomness
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.eps_min = eps_min
        self.Q = {}  # (state_key, action) -> Q‑value

    @staticmethod
    def _state_key(board):
        return tuple(board.flatten())

    @staticmethod
    def _flip_state_action(state_key, action):
        arr = np.array(state_key).reshape(6, 7)
        flipped = np.fliplr(arr)
        flipped_key = tuple(flipped.flatten())
        flipped_action = 8 - action
        return flipped_key, flipped_action

    def valid_actions(self, state):
        return [c + 1 for c in range(7) if state[-1, c] == 0]

    def choose_action(self, state, legal=None, player=None):
        if not legal:
            legal = self.valid_actions(state)

        # exploration
        if random.random() < self.epsilon:
            return random.choice(legal)

        # exploitation with random tie‑break
        s_key = self._state_key(state)
        q_vals = [self.Q.get((s_key, a), 0.0) for a in legal]
        max_q = max(q_vals)

        # collect all actions that tie for max
        best_actions = [a for a, q in zip(legal, q_vals) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        s_key = self._state_key(state)
        n_key = self._state_key(next_state)

        q_sa = self.Q.get((s_key, action), 0.0)
        if done:
            q_next_max = 0.0
        else:
            legal_n = self.valid_actions(next_state)
            q_next_max = max((self.Q.get((n_key, a), 0.0) for a in legal_n), default=0.0)


        # reward shaping: +0.2 for creating an “open three,” −0.2 for letting opponent get one
        shaped = self._count_open_threes(next_state, player=1) \
               - self._count_open_threes(state, player=1)
        reward += 0.2 * shaped

        # TD update
        new_q = q_sa + self.alpha * (reward + self.gamma * q_next_max - q_sa)

        self.Q[(s_key, action)] = new_q

        # # symmetry augmentation
        # fk, fa = self._flip_state_action(s_key, action)
        # self.Q[(fk, fa)] = new_q

    def train_self_play(self, env, episodes):
        for ep in range(1, episodes + 1):
            board = env.reset()
            done = False

            while not done:
                old_turn = env.turn

                # 1) canonical board: current player is always +1
                state = board * old_turn

                # 2) pick move
                action = self.choose_action(state)


                # 3) step
                next_board, reward, done = env.step(action)


                # if you just played a winning move, boost that reward
                if done and reward == 1:
                    reward += 1.0

                # check if opponent can win next turn from this new board
                elif not done and self.opponent_winning_next(next_board, -old_turn):
                    reward -= 0.75

                # an immediate two‑move win threat, give a smaller bonus
                elif not done and self.agent_winning_next(next_board, old_turn):
                    reward += 0.5

                # reward from player’s POV
                agent_r = reward * old_turn

                # next state
                next_state = next_board * old_turn

                # update Q
                self.update(state, action, agent_r, next_state, done)

                board = next_board

            # decay exploration
            self.epsilon = max(self.eps_min, self.epsilon_decay * self.epsilon)

            if ep % 1000 == 0:
                print(f"Episode {ep:5d}/{episodes}, epsilon={self.epsilon:.4f}")

    def _count_open_threes(self, board, player):
        cnt = 0
        rows, cols = board.shape
        # horizontal
        for r in range(rows):
            for c in range(cols-3):
                w = board[r, c:c+4]
                if np.count_nonzero(w == player) == 3 and np.count_nonzero(w == 0) == 1:
                    cnt += 1
        # vertical
        for c in range(cols):
            for r in range(rows-3):
                w = board[r:r+4, c]
                if np.count_nonzero(w == player) == 3 and np.count_nonzero(w == 0) == 1:
                    cnt += 1
        # diag down‑right and down‑left
        for r in range(rows-3):
            for c in range(cols-3):
                w1 = board[r:r+4, c:c+4].diagonal()
                w2 = np.fliplr(board[r:r+4, c:c+4]).diagonal()
                for w in (w1, w2):
                    if np.count_nonzero(w == player) == 3 and np.count_nonzero(w == 0) == 1:
                        cnt += 1
        return cnt

    def opponent_winning_next(self, board, player):
        _, cols = board.shape
        for c in range(cols):
            if board[-1, c] != 0: 
                continue
            r = np.argmax(board[:, c] == 0)
            board[r, c] = player
            win = self._is_win(board, r, c, player)
            board[r, c] = 0
            if win:
                return True
        return False

    def agent_winning_next(self, board, player):
        return self._count_open_threes(board, player) > 0

    def _is_win(self, board, row, col, player):
        def count(dr, dc):
            ct = 0
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r,c] == player:
                ct += 1
                r += dr
                c += dc
            return ct

        for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
            if 1 + count(dr,dc) + count(-dr,-dc) >= 4:
                return True
        return False



class QEnv(ConnectFour):
    def reset(self):
        self.__init__()           # re‑initialize
        return self.board.copy()

    def step(self, col):
        prev_board = self.board.copy()
        prev_feats = self._feature_counts(prev_board, self.turn)

        valid = self.get_next_state(col)
        done, winner = self.check_win()

        # base reward
        if not valid:
            base_r = -1.0
        elif done:
            base_r =  +3.0 if winner==1 else -3.0 if winner==-1 else 0.0
        else:
            base_r = 0.0

        # shaping:
        new_board = self.board
        new_feats = self._feature_counts(new_board, -self.turn)  
        # note: after get_next_state we flipped turn, so -self.turn is the move

        # open-threes for you vs opp
        delta_you3  = new_feats[0] - prev_feats[0]
        delta_opp3  = new_feats[1] - prev_feats[1]
        delta_center = new_feats[2] - prev_feats[2]


        w1 = 0.5 # reward for increasing open threes
        w2 = -0.5 # penalty for opponent's open threes
        w3 = 0.1  # reward for occupying center
        shaped = w1*delta_you3 + w2*delta_opp3 + w3*delta_center


        total_r = base_r + shaped
        return new_board.copy(), total_r, done


    def _feature_counts(self, board, player):
        # returns (open3_you, open3_opp, center_diff)
        you_3 = self.count_open_n(board, player, 3)
        opp_3 = self.count_open_n(board, -player, 3)
        center = board[:, 7//2]
        center_diff = int((center==player).sum() - (center==-player).sum())
        return you_3, opp_3, center_diff


    def get_valid_moves(self, board):
        # return the list of 1-indexed columns that aren’t full yet
        return [c+1 for c in range(7) if board[5, c] == 0]

    def count_open_n(self, b, pl, n):
        count = 0
        # horizontal
        ROWS = 6
        COLS = 7
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



def evaluate_agents(agent, env, uct, games=100):
    import matplotlib.pyplot as plt

    wins_q = 0
    wins_uct = 0
    ties = 0

    for g in range(games):
        game = ConnectFour()

        while True:
            done, winner = game.check_win()
            if done:
                if winner == 1:
                    wins_q += 1
                elif winner == -1:
                    wins_uct += 1
                else:
                    ties += 1
                break

            if game.turn == -1:
                root = Node(parent=None, board=game.board, turn=-1)
                move = uct.search(root)
                if move == (-1, -1):
                    break
                game.board[move] = -1
                game.switch_turn()
            else:
                col = agent.choose_action(game.board, env.get_valid_moves(game.board), game.turn)
                valid = game.get_next_state(col)
                if not valid:
                    # if invalid move, count as a loss
                    wins_uct += 1
                    break
        if g % 10 == 0:
            print(f"{g} amount of games evaluated")

    print(f"Q Agent Wins: {wins_q}")
    print(f"UCT Wins: {wins_uct}")
    print(f"Ties: {ties}")

    # plot
    labels = ['Q Agent Wins', 'UCT Wins', 'Ties']
    values = [wins_q, wins_uct, ties]
    plt.bar(labels, values)
    plt.title(f'Q Agent vs UCT over {games} Games')
    plt.ylabel('Number of Games')
    plt.show()

if __name__ == "__main__":
    from connect_feature_q import FeatureQAgent

    # uct algorithm with 10k simulations
    monte_carlo = UCT(10000)

    player = False # if you decide to play aganst the uct algorithm

    if not player:
        env = QEnv()

        # create Q agent || this agent kinda stinks
        # agent = QLearningAgent()
 
        agent = FeatureQAgent()

        agent.train_self_play(env, episodes = 10000)
        agent.epsilon = 0.05
        agent.eps_min = 0.05


    # uncomment to determine the amount of games and evaluate the agents together
    evaluate_agents(agent, env, monte_carlo, games=100)
    sys.exit()



    # Begin new game (infinite loop, no data tracking)
    while True:
        # Game loop
        game = ConnectFour()

        while True:
            game.print_board()

            # Check game over
            game_over, winner_id = game.check_win()
            if game_over is True:
                if winner_id is None:
                    print("\n\nTIE!!!")
                elif winner_id == -1:
                    print("uct algorithm wins")
                else:
                    print("\n\nq agent wins")
                break
            
            # uct
            if game.turn == -1:
                # create root of tree
                root = Node(parent=None,board=game.board,turn=-1)

                # search the tree and find best move
                mcts_move = monte_carlo.search(root)

                # apply move
                game.board[mcts_move] = -1
                game.switch_turn()

            # player/agent turn
            else:
                if player:
                    col = game.get_next_state()
                else:
                    # agent.choose_action expects (board, valid_actions, player)
                    col = agent.choose_action(game.board, env.get_valid_moves(game.board), game.turn)
                    print(f"Agent selects column {col}")
                    game.get_next_state(col, print_move=True)
                    time.sleep(1)




        continue_game = True
        if not continue_game:
            break
        time.sleep(1)



