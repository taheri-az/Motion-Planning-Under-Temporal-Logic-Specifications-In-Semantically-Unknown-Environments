import itertools
import numpy as np
from collections import deque


def make_check_label_l(regions, true_locations):
    # returns check(state) which gives the true label string at that cell.
    # cells not in true_locations have all atoms false.
    active_by_state = {state: set(props) for state, props in true_locations.items()}

    def check(state):
        active = active_by_state.get(state, set())
        return ' && '.join(r if r in active else f'!{r}' for r in regions)

    return check


def get_states_within_h_distance(m, n, current_state, h):
    def state_to_row_col(state):
        return divmod(state, n)

    def row_col_to_state(row, col):
        if 0 <= row < m and 0 <= col < n:
            return row * n + col
        return None

    def get_adjacent_states(state):
        row, col = state_to_row_col(state)
        adjacent_states = []
        for r, c in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            adjacent_state = row_col_to_state(r, c)
            if adjacent_state is not None:
                adjacent_states.append(adjacent_state)
        return adjacent_states

    visited = set()
    queue = deque([(current_state, 0)])
    while queue:
        state, distance = queue.popleft()
        if distance > h:
            continue
        if state in visited:
            continue
        visited.add(state)
        for next_state in get_adjacent_states(state):
            if next_state not in visited:
                queue.append((next_state, distance + 1))

    return list(visited)


def assign_probabilities_g3(n, m, regions, initial_belief=None):
    # builds the belief grid. keys can be short ('a' or 'a & b') or the full
    # form ('a && !b && !c && !d'). leftover mass goes to "nothing holds",
    # then we normalize. cells not in initial_belief get nothing-holds = 1.
    labels = [
        ' && '.join(regions[i] if bits[i] else f'!{regions[i]}' for i in range(len(regions)))
        for bits in itertools.product([0, 1], repeat=len(regions))
    ]
    label_to_index = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)
    region_set = set(regions)
    empty_label = ' && '.join(f'!{r}' for r in regions)
    empty_idx = label_to_index[empty_label]

    def parse_key(key):
        # turn a user key into the full conjunction string in regions order
        s = key.strip() if isinstance(key, str) else ''
        true_set = set()
        if s:
            for tok in (t.strip() for t in s.replace('&&', '&').split('&')):
                if not tok:
                    continue
                if tok.startswith('!'):
                    prop = tok[1:].strip()
                    if prop not in region_set:
                        raise ValueError(f"Unknown proposition '{prop}' in belief key '{key}'")
                else:
                    if tok not in region_set:
                        raise ValueError(f"Unknown proposition '{tok}' in belief key '{key}'")
                    true_set.add(tok)
        return ' && '.join(r if r in true_set else f'!{r}' for r in regions)

    grid = np.empty((n * m, num_labels), dtype=object)
    for state in range(n * m):
        probs = np.zeros(num_labels, dtype=float)
        if initial_belief is not None and state in initial_belief:
            for key, p in initial_belief[state].items():
                probs[label_to_index[parse_key(key)]] += float(p)
            remainder = 1.0 - probs.sum()
            if remainder > 0:
                probs[empty_idx] += remainder
        else:
            probs[empty_idx] = 1.0

        total = probs.sum()
        if total > 0:
            probs /= total

        grid[state] = [(float(probs[j]), labels[j]) for j in range(num_labels)]
    return grid


def update(grid, state, label):
    label_numbers = len(grid[0])
    for i, (probability, lbl) in enumerate(grid[state]):
        if lbl == label:
            grid[state][i] = (1, lbl)
        else:
            grid[state][i] = (0 / (label_numbers - 1), lbl)
    return grid
