# 15-Puzzle

This project implements an **A\*** search algorithm to solve the classic **15-puzzle** using two different heuristics.

## Features

- Random puzzle generation with 21–42 legal moves.
- A\* search with:
  - **Manhattan Distance** heuristic
  - **Misplaced Tiles** heuristic
- Step-by-step solution visualization in the terminal.
- Cross-platform terminal clearing.

## Heuristics

### 1. Manhattan Distance
Measures the total number of moves each tile is away from its goal position:
- Accurate and admissible.
- Considers both row and column differences.
- Usually leads to faster solutions.

### 2. Misplaced Tiles
Counts how many tiles are not in their correct positions:
- Simpler and faster to compute.
- Less precise than Manhattan.
- May explore more states.

## How It Works

1. A random initial state is generated from a solved puzzle.
2. A\* runs with both heuristics separately.
3. The number of visited states and solution length are displayed.
4. Visualization shows the step-by-step solving process.

## Requirements

- Python 3+
- Runs in any standard terminal (Windows/Linux/macOS)

## Run

```bash
python Task01.py
