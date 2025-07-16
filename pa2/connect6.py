import numpy as np
import math

class ConnectFour:
    def __init__(self, player):
        self.player = player  # current player: -1 or 1
        self.board = np.zeros((6, 7), dtype=int)

    @staticmethod
    def check_win(board, check_tie = True):
        if board is None:
            print("no board provided in check_win")
        rows, cols = board.shape

        for r in range(rows):
            for c in range(cols):
                val = board[r][c]
                if val == 0:
                    continue

                # horizontal
                if c <= cols - 4 and all(board[r][c + i] == val for i in range(4)):
                    return True, val

                # vertical
                if r <= rows - 4 and all(board[r + i][c] == val for i in range(4)):
                    return True, val

                # diagonal 1
                if r <= rows - 4 and c <= cols - 4 and all(board[r + i][c + i] == val for i in range(4)):
                    return True, val

                # diagonal 2
                if r <= rows - 4 and c >= 3 and all(board[r + i][c - i] == val for i in range(4)):
                    return True, val

        if check_tie:
            if np.all(board != 0):
                return True, None

        return False, None


    def get_next_state(self, action):
        '''
        state = current board
        action = column index (0-6)
        player = current player (1 or -1)

        find lowest empty row in the column, put the player in that position, return board and the row it is on
        '''
        empty_rows = np.where(self.board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return False  # or return None
        row = np.max(np.where(self.board[:, action] == 0))
        self.board[row, action] = self.player

        self.switch()
        return True

    def switch(self):
        # switches player
        self.player *= -1

    def print_board(self):
        disp = {0: '.', 1: 'Y', -1: 'R'}
        for row in self.board:
            print(' '.join(disp[int(cell)] for cell in row))
        print('1 2 3 4 5 6 7\n')



class Node:
    def __init__(self, parent, board, player):
        self.value_sum = 0   # value of this node
        self.visit_count = 0 # amount of times visited
        self.parent = parent # parent of this node
        self.board = board   # board
        self.player = -1 if player == 1 else 1 # set root as other turn

        self.children = []   # list of children
        self.terminal = self.check_terminal()
        self.expanded = False

    def check_terminal(self):
        win, _ = ConnectFour.check_win(self.board)
        if win:
            return True
        return False

    def compare_children(self, new_child, children):
        return any(np.array_equal(new_child, child) for child in children)

    def add_child(self):
        # check if node is already expanded
        if self.expanded:
            return None
        
        # time to expand
        existing_boards = [child.board for child in self.children]

        legal_cols = np.where(self.board[5, :] == 0)[0]
        if legal_cols.size == 0:
            # no moves left
            self.expanded = True
            return

        # for each legal col, find the lowest empty row in one vectorized step
        rows = np.argmax(self.board[:, legal_cols] == 0, axis=0)

        # try them in column order
        for idx, col in enumerate(legal_cols):
            row = int(rows[idx])
            tmp = self.board.copy()
            tmp[row, col] = self.player        # place ±1
            # if this position hasn’t already been expanded...
            if not existing_boards or not self.compare_children(tmp, existing_boards):
                # pass current player so __init__ flips for the next node
                self.children.append(Node(self, tmp, self.player))
                return

        # all legal moves were duplicates → fully expanded
        self.expanded = True

class UCT:
    def __init__(self, count):
        #self.game = game
        self.simulation_count = count

    def search(self, node):
        for _ in range(self.simulation_count):
            # select and expand
            node = self.select(node)
            if node is None:
                return (-1, -1)
            
            # simulate
            rollout_winner = self.rollout(node)

            # backprop
            self.bp(node, rollout_winner)

        # get the best chld based on our current state
        max_visit = 0
        best_node = None
        for child in node.children:
            if child.visit_count > max_visit:
                max_visit = child.visit_count
                best_node = child

        if best_node is None:
            return (-1, -1) # ????
        
        for j in range(6):
            for i in range(7):
                if best_node.board[j][i] != node.board[j][i]:
                    return (j, i)
        

    def select(self, node):
        """Selection and expansion phase of MCTS."""
        # Traverse down the tree while node is fully expanded and not terminal
        while len(node.children) == np.count_nonzero(node.board[5] == 0) and not node.terminal:
            # Select child with highest UCT score
            node = self.select_ucb(node)
            if node.terminal:
                return node

        # If terminal, return as leaf
        if node.terminal:
            return node

        # Expand a child
        node.add_child()

        # Return the first unvisited child
        return next((child for child in node.children if child.visit_count == 0), node)

    def select_ucb(self, node):
        c = math.sqrt(2)

        best_score = -np.inf
        best_child = None

        # look at every child of this node
        for child in node.children:
            # the math
            q = child.value_sum / child.visit_count
            ucb = q + c * np.sqrt(np.log(node.visit_count) / child.visit_count)

            # replace values if they are better
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        if best_child is None:
            return node
        return best_child




    def expand(self, node):
        legal_moves = np.count_nonzero(node.board[5] == 0)
        # if num children < number of possible moves, its not expanded
        if len(node.children) < legal_moves:
            return False
        # check if all children have been visited
        visits = np.array([child.visit_count for child in node.children])
        return np.all(visits > 0)

    def pick_unvisited(self, children):
        for child in children:
            if child.visit_count == 0:
                return child
        return None
    
    def rollout(self, node):
        # do rollout on a leaf node
        board = node.board
        turn = node.player

        if node.terminal:
            return self.result(board)

        while True:
            turn *= -1
            moves = self.get_valid_moves(board)
            if moves:
                # select next board randomly
                board = np.random.choice(moves)
                # check if this board is terminal
                terminal = self.result(board)
                if terminal != 0:
                    return terminal
                
            else:
                return self.result(board)


    def get_valid_moves(self, board, player):
        moves = []
        # Get indices of non-full columns
        legal_cols = np.where(board[5, :] == 0)[0]

        # Determine player value
        player_val = 1 if player == -1 else -1

        # For each legal column, find the lowest available row and place the piece
        for col in legal_cols:
            row = np.argmax(board[:, col] == 0)
            tmp = board.copy()
            tmp[row, col] = player_val
            moves.append(tmp)

        return moves

    def result(self, board):
        res, winner = ConnectFour.check_win(board, check_tie= False)
        if res:
            return winner
        return None
    
    def bp(self, node, winner):
        if node.player == winner:
            node.value_sum += 1
        node.visit_count += 1
        if node.parent is None:
            return
        self.bp(node.parent, winner)

if __name__ == "__main__":
    game = ConnectFour(1)
    algorithm = UCT(1000)
    while True:
        game.print_board()

        game_over, winner = game.check_win(game.board)
        if game_over:
            print(winner)
            break
        
        if game.player == 1: # player turn
            move = int(input("Input move: "))
        else:
            root = Node(parent = None, board = game.board, turn = 1)
            move = algorithm.search(root)

        game.get_next_state(move)

        game.switch()
