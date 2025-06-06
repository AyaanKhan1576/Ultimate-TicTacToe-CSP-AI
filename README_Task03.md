# Ultimate Tic-Tac-Toe CSP-AI

## Overview
The program implements an AI for Ultimate Tic-Tac-Toe using PyQt6 for the graphical interface. It includes multiple AI agents with varying strategies and heuristics to play and win the game efficiently.

## Problem Description
Ultimate Tic-Tac-Toe is a strategic extension of the classic Tic-Tac-Toe game, played on a 3×3 grid of smaller 3×3 Tic-Tac-Toe boards. The objective is to win three small boards in a row (horizontally, vertically, or diagonally) on the larger board.

### Rules
- **Game Start:** Begins with an empty 3×3 grid of smaller boards.
- **Move Rules:** The active small board for the next move is determined by the position of the last move.
- **Winning Conditions:**
  - A player wins a small board by forming a horizontal, vertical, or diagonal sequence.
  - A player wins the game by winning three small boards in a row.

## Features
- **Graphical Interface:** Built using PyQt6 for an interactive experience.
- **AI Agents:** Includes `BestAI`, `BasicMinimaxAI`, `LCV_AI`, and `DegreeConstraintAI`.
- **CSP Solver:** Implements Constraint Satisfaction Problem (CSP) formulation with:
  - Forward Checking.
  - Arc Consistency (AC-3).
- **Hybrid Solver:** Combines Alpha-Beta Pruning with CSP for optimal moves.

## Tools and Frameworks
- **PyQt6:** For GUI development.
- **Python Standard Library:** For mathematical operations and system handling.

## How to Run
1. Install PyQt6 using pip:
   ```powershell
   pip install PyQt6
   ```
2. Run the program using the following command:
   ```powershell
   python main.py
   ```
3. Interact with the GUI to play the game or test AI strategies.

## Implementation Details
- **Constraint Formulation:**
  - Variables: Represent possible moves on the 3×3 small boards.
  - Domains: Available moves (X, O, or empty).
  - Constraints: Valid moves, winning conditions, and board state rules.
- **AI Strategies:**
  - Minimum Remaining Values (MRV) heuristic.
  - Constraint Optimization for faster wins.

## Example
Gameplay:
```
Player X places a mark in position (1, 1) of the small board.
Player O must play in position (1, 1) of the large board.
```
