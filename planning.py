import re
import copy
import random
import numpy as np
from collections import defaultdict


def build_transition_map(transitions):
    transition_map = defaultdict(list)
    for j in transitions:
        start_state = j[0]
        next_phy = j[1][0]
        transition_map[(start_state, next_phy)].append(j)
    return transition_map


def get_next_state(m, n, z, action, adj_matrix):
    valid_actions = get_valid_actions(m, n, z, adj_matrix)

    if action not in valid_actions:
        return None

    action_effects = {
        'left': -1,
        'right': 1,
        'up': -n,
        'down': n,
        'stay': 0,
    }

    delta = action_effects[action]
    next_state = z + delta
    return next_state


def PA_values(m, n, product_nodes, adj_matrix):
    value_table = {}
    for PA_nodes in product_nodes:
        value_table[PA_nodes] = {}
        valid_actions = get_valid_actions(m, n, PA_nodes[0], adj_matrix)
        for action in valid_actions:
            value_table[PA_nodes][action] = 0.0
    for PA_nodes in product_nodes:
        valid_actions = get_valid_actions(m, n, PA_nodes[0], adj_matrix)
        for action in valid_actions:
            if PA_nodes[1] == 'accept_all' or PA_nodes[1] == 'Trash':
                value_table[PA_nodes][action] = 0
    return value_table


def evaluate_label_expression(label_expr, true_label_expr):
    label_conditions = re.split(r'\s*&&\s*', label_expr)
    true_label_conditions = re.split(r'\s*&&\s*', true_label_expr)

    label_set = set()
    true_label_set = set()
    for cond in label_conditions:
        negated = cond.startswith('!')
        label = cond[1:] if negated else cond
        label_set.add((label, negated))

    for cond in true_label_conditions:
        negated = cond.startswith('!')
        label = cond[1:] if negated else cond
        true_label_set.add((label, negated))

    return label_set == true_label_set


def update_trigger(states, true_labels, state_info):
    num_states = len(states)
    num_labels = max(len(info) for info in state_info.values())

    matrix = np.zeros((num_states, num_labels))

    for i, state in enumerate(states):
        for j, (probability, label_expr) in enumerate(state_info[state]):
            is_true_label = evaluate_label_expression(label_expr, true_labels[state])
            matrix[i, j] = abs(1 - probability) if is_true_label else abs(0 - probability)

    inf_norm = np.linalg.norm(matrix, np.inf)
    return inf_norm


def get_valid_actions(m, n, z, adj_matrix):
    row, col = divmod(z, n)
    valid_actions = []

    def state_num(row, col):
        return row * n + col

    if row > 0 and adj_matrix[z][state_num(row - 1, col)] > 0:
        valid_actions.append('up')
    if row < m - 1 and adj_matrix[z][state_num(row + 1, col)] > 0:
        valid_actions.append('down')
    if col > 0 and adj_matrix[z][state_num(row, col - 1)] > 0:
        valid_actions.append('left')
    if col < n - 1 and adj_matrix[z][state_num(row, col + 1)] > 0:
        valid_actions.append('right')

    valid_actions.append('stay')
    return valid_actions


def Value_iteration(m, n, value_table, transition_dict, transitions, product_nodes, gamma, adj_matrix, epsilon):
    a = 1
    b = 0
    r = 1
    c = -1
    transition_map = build_transition_map(transitions)
    max_values = {}
    max_actions = {}
    value_table_2 = copy.deepcopy(value_table)

    iteration = 0
    while True:
        delta = 0

        for state, actions in value_table_2.items():
            max_value = max(actions.values())
            max_action = max(actions, key=actions.get)
            max_actions[state] = max_action
            max_values[state] = max_value

        for i in value_table_2:
            if i[1] != 'accept_all' and i[1] != 'Trash':
                current_physical_state = i[0]
                valid_actions = get_valid_actions(m, n, current_physical_state, adj_matrix)

                for action in valid_actions:
                    expected_reward = 0
                    next_physical_state = get_next_state(m, n, current_physical_state, action, adj_matrix)

                    if next_physical_state is not None:
                        possible_transitions = transition_map.get((i, next_physical_state), [])

                        for j in possible_transitions:
                            first_st = str(j[0])
                            second_st = str(j[1])
                            prob = transition_dict[(first_st, second_st)]
                            phy_transition_value = adj_matrix[current_physical_state][next_physical_state]

                            if j[1][1] == 'accept_all':
                                reward = a * c + b * r
                                partial_reward = prob * (
                                    (phy_transition_value * reward)
                                    + (1 - phy_transition_value) * value_table_2[j[0]]['stay']
                                )
                            elif j[1][1] == 'Trash':
                                reward = a * c / (1 - gamma)
                                partial_reward = prob * (
                                    (phy_transition_value * reward)
                                    + (1 - phy_transition_value) * value_table_2[j[0]]['stay']
                                )
                            else:
                                reward = a * c
                                partial_reward = prob * (
                                    (phy_transition_value * (reward + max_values[j[1]] * gamma))
                                    + (1 - phy_transition_value) * value_table_2[j[0]]['stay']
                                )

                            expected_reward += partial_reward

                    delta = max(delta, abs(value_table_2[i][action] - expected_reward))
                    value_table_2[i][action] = expected_reward

        if delta < epsilon:
            break
        iteration += 1

    for state, actions in value_table_2.items():
        max_value = max(actions.values())
        keys_with_max_value = [k for k, v in actions.items() if v == max_value]
        max_action = random.choice(keys_with_max_value)
        max_actions[state] = max_action

    for state, actions in value_table_2.items():
        max_value = max(actions.values())
        max_values[state] = max_value

    return max_actions, max_values
