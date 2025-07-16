import time    
import sys
import random

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent 
        self.position = position # Stored as tuple

        self.g = 0 # cost from start to this node
        self.h = 0 # heuristic of this node
        self.f = 0 # total cost of self.g + self.h

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end, h):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    start_time = time.time()

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # keep track of amoutn of nodes created
    created_nodes = 0

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            cost = current_node.g 
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            runtime = (time.time() - start_time) * 1000
            return path[::-1], cost, created_nodes, runtime # Return reversed path, cost, runtime, nodes created


        # --------------- Restricting movement to four moves, removed 4 tuples ---------------
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) - 1) or node_position[1] < 0:
                continue

            # --------------- Define wall --------------------
            wall = 0
            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] == wall:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            heuristics = {1: one, 2: two, 3: three, 4: four}
            heuristic_function = heuristics[h]

            # Create the f, g, and h values
            new_node.g = current_node.g + maze[node_position[0]][node_position[1]]
            new_node.h = heuristic_function(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h

            # skip if in closed list
            if any(closed_node.position == new_node.position for closed_node in closed_list):
                continue

            # new node already  in the open list
            for open_node in open_list:
                if new_node == open_node and new_node.g >= open_node.g:
                    break
            else:
                # add to open list
                open_list.append(new_node)
                created_nodes += 1

    # no path found
    runtime = (time.time() - start_time) * 1000
    return -1, None, created_nodes, runtime


def one(position, goal):
    return 0

def two(position, goal):
    # manhattan distance heuristic
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def three(position, goal):
    return (abs(position[0] - goal[0]) + abs(position[1] - goal[1]) * 1.1)

def four(position, goal):
    random_error = random.choice([-3,-2,-1,1,2,3])
    h = two(position, goal) * random_error
    return max(h, 0)

# read file for ez inputs
def thing(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    # get start and goal
    start = tuple(map(int, lines[0].strip().split(": ")[1].split(",")))
    goal = tuple(map(int, lines[1].strip().split(": ")[1].split(",")))
    
    # make 2d list
    grid = [list(map(int, line.strip().split())) for line in lines[2:]]
    
    return start, goal, grid

def print_list(l):
    for i in l:
        print(i)

def run_tests(maps):
    heuristics = {1: "H1 (All Zeros)", 2: "H2 (Manhattan)", 3: "H3 (Modified Manhattan)", 4: "H4 (Manhattan with Error)"}

    results = {}
    # amount of times to loop through maps (average results)
    loop = 1

    # Loop over all maps
    for map_number, map_data in maps.items():
        start = map_data["Start"]
        end = map_data["Goal"]
        maze = map_data["maze"]

        # Loop over all heuristics
        for h in range(1, 5):
            path, cost, created_nodes, runtime = astar(maze, start, end, h)
            total_cost = 0
            total_nodes = 0
            total_runtime = 0
            valid_runs = 0  # Keep track of successful runs (if path = -1)
            
            for _ in range(loop):
                path, cost, created_nodes, runtime = astar(maze, start, end, h)

                # If no path found, mark as N/A
                if path == -1:
                    continue

                total_cost += cost
                total_nodes += created_nodes
                total_runtime += runtime
                valid_runs += 1

            # Calculate averages (if valid runs exist)
            avg_cost = total_cost / valid_runs if valid_runs > 0 else "N/A"
            avg_nodes = total_nodes / valid_runs if valid_runs > 0 else "N/A"
            avg_runtime = total_runtime / valid_runs if valid_runs > 0 else "N/A"

            results[(map_number, h)] = [map_number, heuristics[h], avg_cost, avg_nodes, avg_runtime]

    # FANCY TABLE
    print("\n" + "=" * 70)
    print(f"{'Map #':<6} {'Heuristic':<30} {'Avg Path Cost':<15} {'Avg Nodes Created':<18} {'Avg Runtime (ms)':<15}")
    print("=" * 70)

    for row in results.values():
        print(f"{row[0]:<6} {row[1]:<30} {row[2]:<15} {row[3]:<18} {row[4]:<15}")

    print("=" * 70)

def main():
    # get info from coordinates.txt
    start, end, maze = thing("coordinates.txt")
    # Define the maps, 3 given, 2 generated
    map = {
        1:{
            "Start": (1, 2),
            "Goal": (4, 3),
            "maze": [
                [2, 4, 2, 1, 4, 5, 2],
                [0, 1, 2, 3, 5, 3, 1],
                [2, 0, 4, 4, 1, 2, 4],
                [2, 5, 5, 3, 2, 0, 1],
                [4, 3, 3, 2, 1, 0, 1]
            ]
        },
        2: {
            "Start": (3,6),
            "Goal": (5,1),
            "maze": [
                [1 ,3, 2 ,5 ,1, 4, 3],
                [2 ,1 ,3 ,1, 3 ,2 ,5],
                [3 ,0 ,5 ,0, 1 ,2, 2],
                [5 ,3 ,2 ,1, 5 ,0, 3],
                [2, 4 ,1 ,0 ,0 ,2, 0],
                [4, 0 ,2 ,1 ,5 ,3, 4],
                [1 ,5, 1, 0, 2 ,4 ,1]
            ]
        },
        3: {
            "Start": (1,2),
            "Goal": (8,8),
            "maze": [
                [2, 0, 2, 0, 2, 0, 0, 2, 2, 0],
                [1, 2, 3, 5, 2, 1, 2, 5, 1, 2],
                [2, 0, 2, 2, 1, 2, 1, 2, 4, 2],
                [2, 0, 1, 0, 1, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 5, 0, 3, 2, 2, 2],
                [2, 2, 2, 2, 1, 0, 1, 2, 1, 0],
                [1, 0, 2, 1, 3, 1, 4, 3, 0, 1],
                [2, 0, 5, 1, 5, 2, 1, 2, 4, 1],
                [1, 2, 2, 2, 0, 2, 0, 1, 1, 0],
                [5, 1, 2, 1, 1, 1, 2, 0, 1, 2]
            ]
        },
        4:{
            "Start": (1, 3),
            "Goal": (0, 2),
            "maze": [
                [4, 2, 3, 5],
                [4, 1, 3, 2],
                [5, 2, 1, 3],
                [2, 1, 0, 3]
            ]
        },
        5:{
            "Start": (19, 0),
            "Goal": (10, 7),
            "maze": [
                [4, 5, 0, 2, 2, 3, 4, 3, 0, 0, 5, 5, 2, 2, 0, 5, 5, 4, 2, 4],
                [5, 5, 1, 1, 2, 2, 0, 4, 2, 0, 2, 2, 3, 3, 0, 1, 2, 4, 4, 1],
                [1, 5, 0, 2, 0, 2, 5, 4, 4, 2, 2, 4, 2, 5, 1, 1, 3, 2, 3, 3],
                [1, 1, 3, 2, 2, 2, 5, 0, 5, 5, 1, 0, 3, 1, 5, 1, 1, 2, 5, 4],
                [5, 2, 4, 5, 3, 3, 4, 3, 5, 3, 1, 2, 1, 4, 3, 2, 3, 0, 1, 1],
                [3, 0, 0, 2, 5, 4, 5, 5, 4, 1, 4, 3, 1, 4, 4, 0, 3, 0, 3, 1],
                [0, 2, 5, 3, 5, 3, 3, 4, 3, 1, 4, 3, 4, 3, 5, 2, 5, 1, 1, 0],
                [1, 2, 2, 5, 5, 4, 0, 5, 4, 3, 2, 0, 3, 0, 1, 4, 5, 1, 5, 2],
                [4, 1, 2, 0, 5, 3, 3, 3, 5, 2, 4, 3, 3, 4, 2, 0, 5, 2, 4, 5],
                [5, 4, 1, 1, 2, 3, 4, 4, 3, 2, 5, 5, 4, 4, 5, 4, 2, 4, 3, 1],
                [4, 2, 5, 0, 2, 0, 4, 3, 1, 5, 0, 0, 0, 0, 3, 0, 2, 0, 5, 1],
                [0, 5, 0, 1, 3, 2, 0, 2, 5, 4, 3, 2, 0, 5, 4, 1, 1, 3, 5, 1],
                [0, 1, 4, 0, 2, 0, 5, 0, 0, 5, 3, 3, 0, 4, 2, 1, 1, 5, 2, 0],
                [1, 2, 5, 5, 3, 5, 4, 0, 0, 4, 3, 5, 1, 2, 3, 0, 3, 1, 2, 4],
                [4, 5, 4, 3, 0, 3, 3, 3, 1, 3, 3, 1, 4, 3, 3, 4, 1, 5, 3, 0],
                [0, 2, 2, 4, 2, 1, 3, 0, 3, 4, 0, 5, 0, 1, 0, 5, 1, 1, 3, 4],
                [5, 3, 5, 3, 2, 4, 1, 0, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0, 2, 5],
                [1, 1, 3, 5, 0, 2, 3, 5, 1, 1, 4, 4, 4, 3, 3, 0, 0, 3, 2, 2],
                [2, 0, 5, 0, 4, 3, 4, 0, 0, 3, 2, 5, 2, 3, 2, 3, 3, 5, 1, 2],
                [0, 0, 4, 1, 4, 5, 1, 3, 2, 5, 1, 1, 5, 5, 2, 5, 0, 2, 2, 4]
            ]
        }
    }

    # Check for user provided arguments
    if len(sys.argv) != 3:
        print("First argument: <map> (1-5)")
        print("Second argument: <heurstic> (1-4)")
        print("ex astar.py 1 3")
        if True: # Set true for testing runtime and such
            run_tests(map)
        sys.exit(1)

    # Make sure they are valid inputs
    try:
        map_number = int(sys.argv[1])
        heuristic_number = int(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be integers.")
        sys.exit(1)

    if not (1 <= map_number <= 5) or not (1 <= heuristic_number <= 4):
        print("Error: Map number must be 1-5, and heuristic number must be 1-4.")
        sys.exit(1)

    # Print some info based on the map
    print(f"Using map {map_number} with heuristic {heuristic_number}:")
    map_info = map[map_number]

    start = map_info["Start"]
    end = map_info["Goal"]
    maze = map_info["maze"]
    print(f"Start: {start}, End: {end}")
    
    # run program
    path, cost, created_nodes, runtime = astar(maze, start, end, heuristic_number)

    # turn into numpy matrix because they make matrices pretty without any effort
    print("Original Maze:")
    print_list(maze)
    print()

    # if path doesn't exist
    if path == -1:
        print(f"Path does not exist\nCreated Nodes = {created_nodes}\nRuntime = {runtime}")
        sys.exit(1)

    else:
        # Visualize path with a 9
        for i in path:
            row, col = i
            maze[row][col] = 9

        print("Movement:")
        print_list(maze)

        print(f"Path = {path}\nCost = {cost}\nCreated Nodes = {created_nodes}\nRuntime = {runtime}")
    


if __name__ == '__main__':
    main()