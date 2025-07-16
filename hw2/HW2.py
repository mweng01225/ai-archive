import random
import math


def init(n, m):
    # n is number of knapsacks
    # m is number of items
    weights = [random.randint(10, 40) for _ in range(n)]  # U[10,40]
    item_values = [(random.randint(5, 20), random.randint(0, 10)) for _ in range(m)]  # (weight, value)
    return weights, item_values

def random_solution(n, m):
    options = list(range(-1, n))
    return [random.choice(options) for _ in range(m)]

def value_of_solution(solution, sack_weight, item_values):
    n = len(sack_weight)    # number of knapsacks
    m = len(solution)       # number of items

    knap_weights = [0] * n
    knap_values = [0] * n

    # put solution items into napsacks
    for i in range(m):
        item = solution[i]
        if item != -1: # filter out all nonplaced items
            weight, value = item_values[i]
            knap_weights[item] += weight
            knap_values[item] += value

    # Check each knapsack, and how much they are all overweight by
    overweight = sum(max(knap_weights[i] - sack_weight[i], 0) for i in range(n))

    if overweight == 0:
        total_value = sum(knap_values)
    else:
        total_value = -overweight

    
    return total_value, knap_weights

def neighbor_solutions(solution, n):
    neighbors = []
    num_items = len(solution)
    options = list(range(-1, n))

    for i in range(num_items):
        #for every item, try another assignment that isn't the current one
        for new_position in options:
            if solution[i] != new_position:
                new_solution = solution[:]
                new_solution[i] = new_position
                neighbors.append((i, new_position, new_solution))
    return neighbors

def hill_climb(n, m, knapsack_weights, item_values, init_solution=None):
    # Generate a solution if not provided
    if init_solution is None:
        solution = random_solution(n, m)
    else:
        solution = init_solution

    logs = []
    # Evaluate current solution
    total_value, knap_weights = value_of_solution(solution, knapsack_weights, item_values)
    logs.append((solution, knap_weights, total_value))

    # search solution space for an improvement
    improvement = True
    while improvement:
        improvement = False
        best_neighbor = None
        best_neighbor_value = total_value

        # get every neighbor
        neighbors = neighbor_solutions(solution, n)

        for _, _, new_solution in neighbors:
            temp_value, temp_weights = value_of_solution(new_solution, knapsack_weights, item_values)
            if temp_value > best_neighbor_value:
                best_neighbor = (new_solution, temp_weights, temp_value)
                best_neighbor_value = temp_value

        if best_neighbor is not None and best_neighbor_value > total_value:
            improvement = True
            # Update values for another loop
            solution, knap_weights, total_value = best_neighbor
            # log the values
            logs.append(best_neighbor)
        
    return logs, solution, total_value

def hill_climb_random_restarts(n, m, knapsack_weights, item_values, restarts):
    best_solution = None
    best_value = float('-inf')
    runs = []

    for _ in range(restarts):
        logs, solution, values = hill_climb(n, m, knapsack_weights, item_values)
        runs.append(logs)

        if values > best_value:
            best_solution = solution
            best_value = values
    
    return runs, best_solution, best_value


# from https://www.geeksforgeeks.org/what-is-tabu-search/
def tabu_search(n, m, knapsack_weights, item_values, tabu_size=100, max_iters=2000):
    # Random solution
    current_solution = random_solution(n, m)
    current_value, current_weights = value_of_solution(current_solution, knapsack_weights, item_values)
    best_solution = current_solution[:]
    best_value = current_value

    # Store solutions
    tabu_list = []

    logs = []
    logs.append((current_solution, current_weights, current_value))
    
    for iteration in range(max_iters):
        # Get neighbors and filter out those that are tabu
        neighbors = neighbor_solutions(current_solution, n)
        best_neighbor = None
        best_neighbor_value = float('-inf')
        best_neighbor_weights = None

        for move_index, new_position, neighbor in neighbors:
            neighbor_key = tuple(neighbor)
            if neighbor_key in tabu_list:
                continue  # Skip tabu neighbor

            neighbor_value, neighbor_weights = value_of_solution(neighbor, knapsack_weights, item_values)
            if neighbor_value > best_neighbor_value:
                best_neighbor = neighbor
                best_neighbor_value = neighbor_value
                best_neighbor_weights = neighbor_weights

        # No non-tabu neighbor found, break out of the loop
        if best_neighbor is None:
            break

        # Update current solution
        current_solution = best_neighbor
        current_value = best_neighbor_value
        current_weights = best_neighbor_weights

        # Add to the list
        logs.append((current_solution, current_weights, current_value))
        tabu_list.append(tuple(current_solution))
        
        # Maintain the tabu list size
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # Update best solution if improvement
        if current_value > best_value:
            best_solution = current_solution[:]
            best_value = current_value

    return logs, best_solution, best_value


def printlogs(logs):
    for index, (solution, weight_distribution, value) in enumerate(logs):
        print(f"Step: {index}")
        #print(f"Solution: {solution}")
        #print(f"Weight Distribution: {weight_distribution}")
        print(f"Value: {value}")
        print()
    print("-----------------------------")

def print_first_and_last_logs(logs):
    if len(logs) < 2:
        return
    _, _, value = logs[0]
    _, _, value2 = logs[len(logs) - 1]

    print(f"Step: 1")
    print(f"Value: {value}")
    print("...")
    print(f"Step: {len(logs)}")
    print(f"Value: {value2}")
    print()

if __name__ == "__main__":
    #part a
    print_everything = True # SET THIS TO TRUE IF YOU WANT TO PRINT ALL ITERATIONS

    n, m = 3, 20  # knapsacks, items
    knapsack_weights, itemvalues = init(n, m)
    for i in range(5):
        print(f" ---- Iteration {i+1} ---- ")
        logs, solution, value = hill_climb(n, m, knapsack_weights, itemvalues)
        if print_everything:
            printlogs(logs)
        else:
            print_first_and_last_logs(logs)

    print("-----------b----------")
    # part b
    n, m = 5, 50
    restarts = 5
    knapsack_weights, itemvalues = init(n, m)
    logs, solution, value = hill_climb_random_restarts(n, m, knapsack_weights, itemvalues, restarts)
    for run, log in enumerate(logs):
        print(f" ---- Run number {run} ---- ")
        if print_everything:
            printlogs(log)
        else:
            print_first_and_last_logs(log)

    print_everything = False
    print("-------ec----------")
    # extra credit:
    for i in range(5):
        print(f" ---- Iteration {i+1} ---- ")
        logs, solution, value = tabu_search(n, m, knapsack_weights, itemvalues)
        if print_everything:
            printlogs(logs)
        else:
            print_first_and_last_logs(logs)

''' references used:
https://www.geeksforgeeks.org/what-is-tabu-search/
https://en.wikipedia.org/wiki/Tabu_search

Tabu search keep tracks of recently visited solutions and it avoids returning to previously explored solutions that are not particularly good. 
It keeps a tabu list that keeps track of solutions from being revisited, this means that the algorithm won't be stuck at a local maxima compared to hill climb
tabu search may also allow bad solutions to escape a local maximum
This takes significantly longer than regular hill climbing but it produces a better result.
'''