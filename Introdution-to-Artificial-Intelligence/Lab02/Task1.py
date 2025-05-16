## TASK 1 


import random
import heapq
from copy import deepcopy
import time
import os
import platform

def flatten(state):
    return tuple(sum(state, []))

def find_blank(state):
    for i, row in enumerate(state):
        for j, val in enumerate(row):
            if val == 0:
                return i, j

def print_state(state):
    for row in state:
        print(' '.join(f"{v:2d}" if v != 0 else " X" for v in row))
    print()

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def generate_scrambled(k=30):
    n = 4
    state = [[n * i + j + 1 for j in range(n)] for i in range(n)]
    state[n - 1][n - 1] = 0
    bi, bj = n - 1, n - 1
    last = None
    for _ in range(k):
        moves = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = bi + di, bj + dj
            if 0 <= ni < n and 0 <= nj < n and (ni, nj) != last:
                moves.append((ni, nj))
        ni, nj = random.choice(moves)
        state[bi][bj], state[ni][nj] = state[ni][nj], state[bi][bj]
        last = (bi, bj)
        bi, bj = ni, nj
    return state

def manhattan(state):
    n = len(state)
    dist = 0
    for i in range(n):
        for j in range(n):
            val = state[i][j]
            if val != 0:
                ti = (val - 1) // n
                tj = (val - 1) % n
                dist += abs(i - ti) + abs(j - tj)
    return dist

def misplaced(state):
    n = len(state)
    count = 0
    for i in range(n):
        for j in range(n):
            val = state[i][j]
            if val != 0 and val != i * n + j + 1:
                count += 1
    return count

def astar(start, heuristic):
    n = len(start)
    goal = tuple(list(range(1, n * n)) + [0])
    start_flat = flatten(start)
    open_set = [(heuristic(start), 0, start_flat, [])]
    visited = {start_flat: 0}
    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current == goal:
            return path + [current], len(visited)
        state = [list(current[i * n:(i + 1) * n]) for i in range(n)]
        bi, bj = find_blank(state)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = bi + di, bj + dj
            if 0 <= ni < n and 0 <= nj < n:
                new_state = deepcopy(state)
                new_state[bi][bj], new_state[ni][nj] = new_state[ni][nj], new_state[bi][bj]
                new_flat = flatten(new_state)
                new_g = g + 1
                if new_flat not in visited or new_g < visited[new_flat]:
                    visited[new_flat] = new_g
                    h = heuristic(new_state)
                    heapq.heappush(open_set, (new_g + h, new_g, new_flat, path + [current]))
    return None, len(visited)

def visualize_solution(path, delay=0.3):
    for flat_state in path:
        clear_screen()
        state = [list(flat_state[i * 4:(i + 1) * 4]) for i in range(4)]
        print_state(state)
        time.sleep(delay)
    print("Solved!")

if __name__ == '__main__':
    k = random.randint(21, 42)
    init = generate_scrambled(k)
    print("Initial state:")
    print_state(init)

    for name, h in [("Manhattan", manhattan), ("Misplaced Tiles", misplaced)]:
        print(f"Heuristic: {name}")
        solution, visited = astar(init, h)
        print(f"Solution length: {len(solution) - 1}")
        input("Press Enter to visualize solution...")
        visualize_solution(solution)
