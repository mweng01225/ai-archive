'''
States:
R = Rested
T = Tired
D = homework Done
U = homework Undone
8p = eight o'clock pm
Actions:
P = Party
R = Rest
S = Study
any means any action has the same effect
note: not all actions are possible in all states

red = rewards
green = transition probabilities
gray = terminal state
'''
# ----------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------- SET part1 / part2 = True/False for printing -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------


transitions = {
    'RU8p': { # Layer 1
        'P': [(1.0,'TU10p', 2)],
        'R': [(1.0,'RU10p', 0)],
        'S': [(1.0,'RD10p', -1)]
    },

    'TU10p':{ # Layer 2 correct
        'P': [(1.0,'RU10a', 2)],
        'R': [(1.0,'RU8a', 0)],
    },
    'RU10p':{
        'R': [(1.0,'RU8a', 0)],
        'P': [(0.5,'RU8a', 2), 
              (0.5,'RU10a', 2)],
        'S': [(1.0,'RD8a', -1)]
    },
    'RD10p':{
        'R': [(1.0,'RD8a', 0)],
        'P': [(0.5,'RD8a', 2),
              (0.5,'RD10a', 2)],
    },

    'RU8a' :{ # Layer 3
        'P': [(1.0,'TU10a', 2)],
        'R': [(1.0,'RU10a', 0)],
        'S': [(1.0,'RD10a', -1)]
    },
    'RD8a' :{
        'R': [(1.0,'RD10a', 0)],
        'P': [(1.0,'TD10a', 2)],
    },

    'TU10a':{ # Layer 4 correct
        'any':[(1.0, 'EXIT', -1)]
    },
    'RU10a':{
        'any':[(1.0, 'EXIT', 0)]
    },
    'RD10a':{
        'any':[(1.0, 'EXIT', 4)]
    },
    'TD10a':{
        'any':[(1.0, 'EXIT', 3)]
    }
}

'''
----- part 1 -------
Implement the value iteration algorithm and use it to find the optimal policy for this MDP. 
- Set all value estimates to 0 initially. 
- Use a discount rate (lambda) of 0.99. 
Each time you update the value of a state, print out the previous value, the new value, the
estimated value of each action, and the action selected. 
- Continue to update each state until the maximum change in the value of any state in a single iteration is less than 0.001. 

At the end, print out the number of iterations (i.e., the number of times you updated each state), 
the final values for each state, and the final optimal policy.
'''
# prints part1 results
part1 = True
if part1:
    import copy
    states = list(transitions.keys()) + ['EXIT'] # exit = terminal state

    # set all values to 0
    V = {s: 0.0 for s in states}

    # discount rate = 0.99
    lambdaa = 0.99

    # Continue to update each state until the maximum change in the value of any state in a single iteration is less than 0.001. 
    theta = 0.001
    iteration = 0

    policy = {s: None for s in states}

    # get the estimated value for each action from a given state
    def one_ahead(state, V):
        action_values = {}
        if state not in transitions:
            return action_values
        
        for action, outcomes in transitions[state].items():
            total = 0
            for probability, next_state, reward in outcomes:
                total += probability * (reward + lambdaa * V[next_state])
            action_values[action] = total
        return action_values

    # iteration loop
    while True:
        # track max change
        delta = 0
        new_V = copy.deepcopy(V)
        print()
        print(f"---------- Iteration {iteration+1} ----------")
        for state in states:
            if state not in transitions:
                continue  # Skip terminal state

            A = one_ahead(state, V)
            if not A:
                continue

            best_action = max(A, key=A.get)
            best_value = A[best_action]
            delta = max(delta, abs(best_value - V[state]))

            # print calculated values
            print()
            print(f"State: {state}")
            print(f"  Prev V: {V[state]:.4f}")
            for a, q in A.items():
                print(f"  Q({state}, {a}) = {q:.4f}")
            print(f"  New V: {best_value:.4f} via action: {best_action}")

            new_V[state] = best_value
            policy[state] = best_action
        V = new_V
        iteration += 1
        # stop loop when changes are small enough
        if delta < theta:
            break 

    print()
    print("---------- Final Values and Policy ----------")
    for state in states:
        print(f"{state}: V = {V[state]:.4f}, Policy = {policy[state]}")
    print()
    print(f"Total iterations: {iteration}")


'''
part 2
Implement Q-learning and use it to find the optimal policy for this MDP. 
Note that for this algorithm you will need the Q values, which are values for state/action pairs.
Similar to before, you will run episodes repeatedly until the maximum change in any Q value is less than 0.001. 
- Use a learning rate (alpha) of 0.1, and a discount rate (lambda) of 0.99. 
- Use an epsilon-greedy policy for training that places 80% of the weight on the optimal action, 
and chooses a uniform random action with the remaining 20% probability (recall that the Q-learning updates and convergence are
independent of the policy being followed, so it should converge as long as every state/action pair continues to be selected).
- Each time you update a Q value, print out the previous value, the new value, the immediate reward, and the Q value for the next state. 
- At the end, print out the number of episodes, the final Q values, and the optimal policy.
'''
# prints part2 results
part2 = True
if part2:
    import random

    # list the states and actions
    states = list(transitions.keys()) + ['EXIT'] # exit = terminal state
    actions = ['P', 'R', 'S', 'any']

    # parameters
    alpha = 0.1     # learning rate
    lambdaa = 0.99  # discount
    epsilon = 0.2   # 80% greedy, 20% explore
    theta = 0.001

    # initialize q table
    Q = { s: { a:0.0 for a in transitions[s].keys() } 
        for s in transitions }  

    max_delta = float('inf')
    episode = 0

    while max_delta > theta:
        state = 'RU8p'  # top layer1 state
        max_delta = 0   
        episode += 1

        while state != 'EXIT':
            # get all valid actions
            valid_actions = list(transitions[state].keys())

            if random.random() < (1 - epsilon): 
                # greedy
                action = max(valid_actions, key=lambda a: Q[state][a])
            else:  
                # explore
                action = random.choice(valid_actions)

            # transition
            outcomes = transitions[state][action]
            probs, next_states, rewards = zip(*outcomes)
            idx = random.choices(range(len(outcomes)), weights=probs)[0]
            next_state, reward = next_states[idx], rewards[idx]

            # q update
            old_q = Q[state][action]
            if next_state in Q:
                next_max = max(Q[next_state].values())
            else:
                next_max = 0.0  # terminal state
            new_q = old_q + alpha * (reward + lambdaa * next_max - old_q)
            Q[state][action] = new_q


            delta = abs(new_q - old_q)
            if delta > max_delta:
                max_delta = delta

            # print the calculated values
            print(f"Episode {episode} || State: {state}, Action: {action}")
            print(f"Reward: {reward}")
            print(f"Old Q: {old_q:.4f}, New Q: {new_q:.4f}, Max Q next: {next_max:.4f}")
            print()

            # go next state
            state = next_state 


    policy = {}
    for state in Q:
        valid_actions = transitions.get(state, {}).keys()
        if valid_actions == {'any'}:
            policy[state] = 'any'
        else:
            policy[state] = max(Q[state], key=Q[state].get)
    policy['EXIT'] = None

    print()
    print("---------- Final Q-values ----------")
    for state in Q:
        for action in Q[state]:
            print(f"Q[{state}][{action}] = {Q[state][action]:.4f}")

    print()
    print("---------- Final Policy ----------")
    for state, action in policy.items():
        if state == 'EXIT':
            continue
        print(f"{state}: {action}")

    print()
    print(f"Total episodes: {episode}")