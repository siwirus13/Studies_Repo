## TASK 1






import random
import heapq
from copy import deepcopy


def flatten(state):
    return sum(state, [])


def find_blank(state):
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j

# Solvability check

def count_inversions(seq):
    inv = 0
    arr = [x for x in seq if x != 0]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]: inv += 1
    return inv


def is_solvable(state):
    flat = flatten(state)
    inv = count_inversions(flat)
    blank_row, _ = find_blank(state)
    # row counting from bottom (1-indexed)
    row_from_bottom = len(state) - blank_row
    if row_from_bottom % 2 == 1:
        return inv % 2 == 0
    else:
        return inv % 2 == 1

# Heuristics definitions

def manhattan(state):
    dist = 0
    n = len(state)
    for i in range(n):
        for j in range(n):
            val = state[i][j]
            if val:
                target_i = (val - 1) // n
                target_j = (val - 1) % n
                dist += abs(i - target_i) + abs(j - target_j)
    return dist


def linear_conflict(state):
    n = len(state)
    base = manhattan(state)
    conflict = 0
    for i in range(n):
        row = state[i]
        for a in range(n):
            for b in range(a + 1, n):
                val_a, val_b = row[a], row[b]
                if val_a and val_b:
                    # both in this row's goal row?
                    goal_row_a = (val_a - 1) // n
                    goal_row_b = (val_b - 1) // n
                    if goal_row_a == goal_row_b == i and val_a > val_b:
                        conflict += 1
    for j in range(n):
        col = [state[i][j] for i in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                val_a, val_b = col[a], col[b]
                if val_a and val_b:
                    goal_col_a = (val_a - 1) % n
                    goal_col_b = (val_b - 1) % n
                    if goal_col_a == goal_col_b == j and val_a > val_b:
                        conflict += 1
    return base + 2 * conflict

# A* Algorithm-

def astar(start, heuristic):
    n = len(start)
    goal = tuple(range(1, n*n)) + (0,)

    start_flat = tuple(flatten(start))
    open_set = [(heuristic(start), 0, start_flat, [])]
    visited = {start_flat: 0}
    count = 0

    while open_set:
        f, g, state_flat, path = heapq.heappop(open_set)
        count += 1
        if state_flat == goal:
            return path, count

        state = [list(state_flat[i*n:(i+1)*n]) for i in range(n)]
        bi, bj = find_blank(state)

        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = bi + di, bj + dj
            if 0 <= ni < n and 0 <= nj < n:
                new_state = deepcopy(state)
                new_state[bi][bj], new_state[ni][nj] = new_state[ni][nj], new_state[bi][bj]
                new_flat = tuple(flatten(new_state))
                new_g = g + 1
                if new_flat not in visited or new_g < visited[new_flat]:
                    visited[new_flat] = new_g
                    h = heuristic(new_state)
                    heapq.heappush(open_set, (new_g + h, new_g, new_flat, path + [new_state[ni][nj]]))
    return None, count

# Solvable start positions generator

def random_solvable(n=4):
    arr = list(range(n*n))
    while True:
        random.shuffle(arr)
        grid = [arr[i*n:(i+1)*n] for i in range(n)]
        # ensure blank at bottom-right
        bi, bj = find_blank(grid)
        grid[bi][bj], grid[n-1][n-1] = grid[n-1][n-1], grid[bi][bj]
        if is_solvable(grid):
            return grid


def print_state(state):
    for row in state:
        print(' '.join(f"{v:2d}" for v in row))

if __name__ == '__main__':
    init = random_solvable()
    print("Initial state:")
    print_state(init)

    for name, h in [("Manhattan", manhattan), ("Linear Conflict", linear_conflict)]:
        print(f"\nHeuristic: {name}")
        path, visited = astar(init, h)
        print("Moves:", path)
        print("Number of visited states:", visited)
