import math
import time
import random 
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QMenuBar, QMenu, QMessageBox, QInputDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout
)
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QAction, QMouseEvent, QFont
from PyQt6.QtCore import Qt, QRectF, QSize, QTimer, pyqtSlot

class SimpleQueue:
    def __init__(self):
        self._items = []
    def enqueue(self, item):
        self._items.append(item)
    def dequeue(self):
        if not self.is_empty():
            return self._items.pop(0)
        return None
    def is_empty(self):
        return len(self._items) == 0
    def __len__(self):
        return len(self._items)
    def __str__(self):
        return f"Queue({self._items})"

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
DRAW = 'D'
BOARD_SIZE = 3
TOTAL_BOARDS = BOARD_SIZE * BOARD_SIZE
TOTAL_CELLS = TOTAL_BOARDS * TOTAL_BOARDS 

# Checks if a 3x3 board segment (list of 9) has a winner.
def check_win(board_segment):
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for line in lines:
        first = board_segment[line[0]]
        if first != EMPTY and all(board_segment[i] == first for i in line):
             return first 
    return None 

# Checks if a 3x3 board segment is full.
def is_full(board_segment):
    return EMPTY not in board_segment


# Valid Moves
def get_valid_moves(main_board, small_board_winners, active_small_board_idx):
    valid_moves = []
    can_play_anywhere = False
    if active_small_board_idx == -1:
        can_play_anywhere = True
    elif active_small_board_idx < 0 or active_small_board_idx >= TOTAL_BOARDS:
         can_play_anywhere = True
    else:
         target_board_status = small_board_winners[active_small_board_idx]
         target_board_full = False
         if target_board_status == EMPTY:
             target_board_full = is_full(main_board[active_small_board_idx])
         if target_board_status != EMPTY or target_board_full:
              can_play_anywhere = True

    if can_play_anywhere:
        for i in range(TOTAL_BOARDS):
            if small_board_winners[i] == EMPTY:
                small_board = main_board[i]
                if not is_full(small_board):
                    for j in range(TOTAL_BOARDS): 
                        if small_board[j] == EMPTY:
                            valid_moves.append((i, j))
    else:
        board_idx = active_small_board_idx
        if small_board_winners[board_idx] == EMPTY:
             small_board = main_board[board_idx]
             if not is_full(small_board):
                 for j in range(TOTAL_BOARDS):
                     if small_board[j] == EMPTY:
                         valid_moves.append((board_idx, j))

    seen = set()
    unique_valid_moves = []
    for item in valid_moves:
        if item not in seen:
            unique_valid_moves.append(item)
            seen.add(item)
    return unique_valid_moves


def deep_copy_board(board):
    new_board = []
    for small_board in board:
        new_board.append(list(small_board)) # Copy inner list
    return new_board

# AI Base Class 
class AIAgent:
    def __init__(self, player, opponent, name="AI"):
        self.player = player
        self.opponent = opponent
        self.name = name
        self.nodes_visited = 0 

    # Abstract
    def get_move(self, main_board, small_board_winners, active_small_board_idx):
        raise NotImplementedError("Each AI agent must implement get_move")

    def _evaluate_small_board(self, board_segment, player_perspective):
        score = 0
        winner = check_win(board_segment)
        opponent_perspective = self.opponent if player_perspective == self.player else self.player
        if winner == player_perspective: return 100
        elif winner == opponent_perspective: return -100
        elif is_full(board_segment): return 0 # Draw is neutral

        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for line in lines:
            player_count, opponent_count, empty_count = 0, 0, 0
            for i in line:
                cell = board_segment[i]
                if cell == player_perspective: player_count += 1
                elif cell == opponent_perspective: opponent_count += 1
                else: empty_count += 1
            if player_count == 2 and empty_count == 1: score += 10
            elif opponent_count == 2 and empty_count == 1: score -= 15
            elif player_count == 1 and empty_count == 2: score += 1
            elif opponent_count == 1 and empty_count == 2: score -= 1
        if board_segment[4] == player_perspective: score += 0.5 # Center control bonus
        elif board_segment[4] == opponent_perspective: score -= 0.5
        return score

    def _evaluate_board(self, main_board, small_board_winners, player_to_move):
        game_winner = check_win(small_board_winners) # Check win on the big board
        if game_winner == self.player: return 10000
        elif game_winner == self.opponent: return -10000

        all_boards_decided = all(s != EMPTY for s in small_board_winners)
        if all_boards_decided and game_winner is None: return 0 # Draw

        total_score = 0
        large_board_eval = self._evaluate_small_board(small_board_winners, self.player)
        total_score += large_board_eval * 10 # Weight large board structure
        for i in range(TOTAL_BOARDS):
            if small_board_winners[i] == EMPTY:
                total_score += self._evaluate_small_board(main_board[i], self.player)
            elif small_board_winners[i] == self.player:
                 total_score += 100 # Keep user's explicit bonus/penalty
            elif small_board_winners[i] == self.opponent:
                 total_score -= 100
        return total_score

# Default AI: Hybrid Solver (Alpha-Beta + CSP + Opt + MRV) 
class BestAI(AIAgent):
    def __init__(self, player, opponent, depth_limit=4, use_mrv=True):
        super().__init__(player, opponent, name=f"BestAI(d={depth_limit}{',MRV' if use_mrv else ''})")
        self.depth_limit = max(1, int(depth_limit))
        self.use_mrv = use_mrv

    def get_move(self, main_board, small_board_winners, active_small_board_idx):
        self.nodes_visited = 0
        start_time = time.time()
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        if not valid_moves: return None, 0.0, 0

        best_move = valid_moves[0]
        ordered_moves = list(valid_moves)
        if self.use_mrv:
            can_play_anywhere = False
            current_forced_board = active_small_board_idx
            if current_forced_board == -1 or \
               (current_forced_board >= 0 and current_forced_board < TOTAL_BOARDS and \
                (small_board_winners[current_forced_board] != EMPTY or is_full(main_board[current_forced_board]))):
                can_play_anywhere = True
            if can_play_anywhere and len(set(m[0] for m in valid_moves)) > 1:
                board_options = {}
                for board_idx in range(TOTAL_BOARDS):
                    if small_board_winners[board_idx] == EMPTY:
                        empty_count = main_board[board_idx].count(EMPTY)
                        if empty_count > 0: board_options[board_idx] = empty_count
                ordered_moves.sort(key=lambda move: board_options.get(move[0], float('inf')))

        best_val, best_move_found = self._minimax_alpha_beta(
            main_board, small_board_winners, active_small_board_idx,
            self.depth_limit, -math.inf, math.inf, True, ordered_moves
        )
        end_time = time.time()
        time_taken = end_time - start_time
        best_move = best_move_found if best_move_found is not None else (valid_moves[0] if valid_moves else None)

        if best_move and best_move not in valid_moves:
             print(f"WARNING: {self.name} chose invalid move {best_move}! Falling back.")
             best_move = valid_moves[0] if valid_moves else None

        return best_move, time_taken, self.nodes_visited

    def _minimax_alpha_beta(self, main_board, small_board_winners, current_active_idx, depth, alpha, beta, is_maximizing_player, ordered_moves_hint=None):
        self.nodes_visited += 1
        game_winner = check_win(small_board_winners)
        all_boards_decided = all(s != EMPTY for s in small_board_winners)
        if game_winner is not None or depth == 0 or all_boards_decided:
            eval_player = self.player
            score = self._evaluate_board(main_board, small_board_winners, eval_player)
            return score, None

        current_player_turn = self.player if is_maximizing_player else self.opponent
        if depth == self.depth_limit and ordered_moves_hint:
             valid_moves = ordered_moves_hint
        else:
             valid_moves = get_valid_moves(main_board, small_board_winners, current_active_idx)

        if not valid_moves:
             eval_player = self.player
             score = self._evaluate_board(main_board, small_board_winners, eval_player)
             return score, None

        best_move_for_level = valid_moves[0]

        if is_maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                board_idx, cell_idx = move
                new_board = deep_copy_board(main_board); new_board[board_idx][cell_idx] = current_player_turn
                new_small_winners = list(small_board_winners)
                small_winner = check_win(new_board[board_idx])
                if small_winner and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = small_winner
                elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = DRAW
                next_active_board = cell_idx
                evaluation, _ = self._minimax_alpha_beta(new_board, new_small_winners, next_active_board, depth - 1, alpha, beta, False)
                if evaluation > max_eval:
                    max_eval = evaluation; best_move_for_level = move
                alpha = max(alpha, evaluation)
                if beta <= alpha: break
            return max_eval, best_move_for_level
        else: 
            min_eval = math.inf
            for move in valid_moves:
                board_idx, cell_idx = move
                new_board = deep_copy_board(main_board); new_board[board_idx][cell_idx] = current_player_turn
                new_small_winners = list(small_board_winners)
                small_winner = check_win(new_board[board_idx])
                if small_winner and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = small_winner
                elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = DRAW
                next_active_board = cell_idx
                evaluation, _ = self._minimax_alpha_beta(new_board, new_small_winners, next_active_board, depth - 1, alpha, beta, True)
                if evaluation < min_eval:
                    min_eval = evaluation; best_move_for_level = move
                beta = min(beta, evaluation)
                if beta <= alpha: break
            return min_eval, best_move_for_level


# AI Variation: Basic Minimax (No Alpha-Beta) 
class BasicMinimaxAI(AIAgent):
    def __init__(self, player, opponent, depth_limit=4): 
         super().__init__(player, opponent, name=f"Minimax(d={depth_limit})")
         self.depth_limit = max(1, int(depth_limit))

    def get_move(self, main_board, small_board_winners, active_small_board_idx):
        self.nodes_visited = 0
        start_time = time.time()
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        if not valid_moves: return None, 0.0, 0

        best_val, best_move_found = self._minimax(
            main_board, small_board_winners, active_small_board_idx, self.depth_limit, True
        )
        end_time = time.time()
        time_taken = end_time - start_time
        best_move = best_move_found if best_move_found is not None else (valid_moves[0] if valid_moves else None)

        if best_move and best_move not in valid_moves:
             print(f"WARNING: {self.name} chose invalid move {best_move}! Falling back.")
             best_move = valid_moves[0] if valid_moves else None

        return best_move, time_taken, self.nodes_visited

    def _minimax(self, main_board, small_board_winners, current_active_idx, depth, is_maximizing_player):
        self.nodes_visited += 1
        game_winner = check_win(small_board_winners)
        all_boards_decided = all(s != EMPTY for s in small_board_winners)
        if game_winner is not None or depth == 0 or all_boards_decided:
            eval_player = self.player
            score = self._evaluate_board(main_board, small_board_winners, eval_player)
            return score, None

        current_player_turn = self.player if is_maximizing_player else self.opponent
        valid_moves = get_valid_moves(main_board, small_board_winners, current_active_idx)
        if not valid_moves:
             eval_player = self.player
             score = self._evaluate_board(main_board, small_board_winners, eval_player)
             return score, None

        best_move_for_level = valid_moves[0]

        if is_maximizing_player:
            max_eval = -math.inf
            for move in valid_moves:
                board_idx, cell_idx = move
                new_board = deep_copy_board(main_board); new_board[board_idx][cell_idx] = current_player_turn
                new_small_winners = list(small_board_winners)
                small_winner = check_win(new_board[board_idx])
                if small_winner and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = small_winner
                elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = DRAW
                next_active_board = cell_idx
                evaluation, _ = self._minimax(new_board, new_small_winners, next_active_board, depth - 1, False)
                if evaluation > max_eval:
                    max_eval = evaluation; best_move_for_level = move
            return max_eval, best_move_for_level
        else: 
            min_eval = math.inf
            for move in valid_moves:
                board_idx, cell_idx = move
                new_board = deep_copy_board(main_board); new_board[board_idx][cell_idx] = current_player_turn
                new_small_winners = list(small_board_winners)
                small_winner = check_win(new_board[board_idx])
                if small_winner and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = small_winner
                elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY: new_small_winners[board_idx] = DRAW
                next_active_board = cell_idx
                evaluation, _ = self._minimax(new_board, new_small_winners, next_active_board, depth - 1, True)
                if evaluation < min_eval:
                    min_eval = evaluation; best_move_for_level = move
            return min_eval, best_move_for_level


# AI Variation: Alpha-Beta + LCV Heuristic 
class LCV_AI(BestAI): # Inherits alpha-beta structure
     # (Keep __init__)
     def __init__(self, player, opponent, depth_limit=4):
        super().__init__(player, opponent, depth_limit=depth_limit, use_mrv=False)
        self.name = f"LCV_AI(d={depth_limit})"

     def get_move(self, main_board, small_board_winners, active_small_board_idx):
        self.nodes_visited = 0
        start_time = time.time()
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        if not valid_moves: return None, 0.0, 0

        move_scores = []
        for move in valid_moves:
            board_idx, cell_idx = move
            temp_board = deep_copy_board(main_board); temp_board[board_idx][cell_idx] = self.player
            temp_small_winners = list(small_board_winners)
            small_winner = check_win(temp_board[board_idx])
            if small_winner and temp_small_winners[board_idx] == EMPTY: temp_small_winners[board_idx] = small_winner
            elif is_full(temp_board[board_idx]) and temp_small_winners[board_idx] == EMPTY: temp_small_winners[board_idx] = DRAW
            next_active_board_for_opponent = cell_idx
            opponent_options = get_valid_moves(temp_board, temp_small_winners, next_active_board_for_opponent)
            move_scores.append((move, len(opponent_options)))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_moves = [score[0] for score in move_scores]

        best_val, best_move_found = self._minimax_alpha_beta(
            main_board, small_board_winners, active_small_board_idx,
            self.depth_limit, -math.inf, math.inf, True, ordered_moves
        )
        end_time = time.time()
        time_taken = end_time - start_time
        best_move = best_move_found if best_move_found is not None else (valid_moves[0] if valid_moves else None)

        if best_move and best_move not in valid_moves:
             print(f"WARNING: {self.name} chose invalid move {best_move}! Falling back.")
             best_move = valid_moves[0] if valid_moves else None

        return best_move, time_taken, self.nodes_visited


# AI Variation: Alpha-Beta + Degree Heuristic 
class DegreeConstraintAI(BestAI): # Inherits alpha-beta structure
     # (Keep __init__)
     def __init__(self, player, opponent, depth_limit=4):
        super().__init__(player, opponent, depth_limit=depth_limit, use_mrv=False)
        self.name = f"DegreeAI(d={depth_limit})"

     def get_move(self, main_board, small_board_winners, active_small_board_idx):
        self.nodes_visited = 0
        start_time = time.time()
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        if not valid_moves: return None, 0.0, 0

        move_scores = []
        for move in valid_moves:
            board_idx, cell_idx = move
            target_board_for_opponent_idx = cell_idx
            opponent_options_count = float('inf')
            if 0 <= target_board_for_opponent_idx < TOTAL_BOARDS:
                opp_target_status = small_board_winners[target_board_for_opponent_idx]
                opp_target_full = is_full(main_board[target_board_for_opponent_idx])
                if opp_target_status == EMPTY and not opp_target_full:
                    opponent_options_count = main_board[target_board_for_opponent_idx].count(EMPTY)
            move_scores.append((move, opponent_options_count))
        move_scores.sort(key=lambda x: x[1])
        ordered_moves = [score[0] for score in move_scores]

        best_val, best_move_found = self._minimax_alpha_beta(
            main_board, small_board_winners, active_small_board_idx,
            self.depth_limit, -math.inf, math.inf, True, ordered_moves
        )
        end_time = time.time()
        time_taken = end_time - start_time
        best_move = best_move_found if best_move_found is not None else (valid_moves[0] if valid_moves else None)

        if best_move and best_move not in valid_moves:
             print(f"WARNING: {self.name} chose invalid move {best_move}! Falling back.")
             best_move = valid_moves[0] if valid_moves else None

        return best_move, time_taken, self.nodes_visited

# Dictionary mapping names to AI factory functions 
AI_CLASSES = {
    "Default - (d=4)": lambda p,o: BestAI(p, o, depth_limit=4),
    "Minimax (d=4)": lambda p,o: BasicMinimaxAI(p, o, depth_limit=4),
    "LCV AI (d=4)": lambda p,o: LCV_AI(p, o, depth_limit=4),
    "Degree AI (d=4)": lambda p,o: DegreeConstraintAI(p, o, depth_limit=4),
}



# CSP Solver Implementation for Ultimate Tic-Tac-Toe
# This implements a Constraint Satisfaction Problem approach to UTTT

class CSPSolver(AIAgent):
    def __init__(self, player, opponent, use_forward_checking=True, use_arc_consistency=True, 
                 use_mrv=True, use_optimization=True, use_alpha_beta=True, depth_limit=4):
        name_parts = ["CSP"]
        if use_forward_checking: name_parts.append("FC")
        if use_arc_consistency: name_parts.append("AC3")
        if use_mrv: name_parts.append("MRV")
        if use_optimization: name_parts.append("OPT")
        if use_alpha_beta: name_parts.append("AB")
        name = f"{'-'.join(name_parts)}(d={depth_limit})"
        super().__init__(player, opponent, name=name)
        
        self.use_forward_checking = use_forward_checking
        self.use_arc_consistency = use_arc_consistency
        self.use_mrv = use_mrv
        self.use_optimization = use_optimization
        self.use_alpha_beta = use_alpha_beta
        self.depth_limit = depth_limit
        
        # For tracking performance
        self.backtracks = 0
        self.arc_checks = 0
    
    def get_move(self, main_board, small_board_winners, active_small_board_idx):
        self.nodes_visited = 0
        self.backtracks = 0
        self.arc_checks = 0
        
        start_time = time.time()
        
        # Get valid moves within constraint satisfaction framework
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        if not valid_moves: 
            return None, 0.0, 0
        
        # Create CSP variables and domains
        csp = self._create_csp(main_board, small_board_winners, active_small_board_idx)
        
        # Order moves according to CSP heuristics
        ordered_moves = self._order_moves_by_heuristics(csp, valid_moves, main_board, small_board_winners)
        
        # Solve using backtracking search with CSP techniques
        best_val, best_move = self._csp_search(
            main_board, small_board_winners, active_small_board_idx,
            self.depth_limit, -math.inf if self.use_alpha_beta else None, 
            math.inf if self.use_alpha_beta else None,
            True, ordered_moves
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Fallback in case our search fails
        best_move = best_move if best_move is not None else (valid_moves[0] if valid_moves else None)
        
        # Validate move
        if best_move and best_move not in valid_moves:
            print(f"WARNING: {self.name} chose invalid move {best_move}! Falling back.")
            best_move = valid_moves[0] if valid_moves else None
        
        stats_output = f"Nodes: {self.nodes_visited}, Backtracks: {self.backtracks}, AC3 Checks: {self.arc_checks}"
        print(f"CSP search stats: {stats_output}")
        
        return best_move, time_taken, self.nodes_visited
    
    def _create_csp(self, main_board, small_board_winners, active_small_board_idx):
        """Create a CSP representation of the current UTTT state"""
        # Create variables and their domains
        variables = {}
        domains = {}
        constraints = {}
        
        # 1. Define variables (one for each empty cell in valid small boards)
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        for move in valid_moves:
            board_idx, cell_idx = move
            var_name = f"b{board_idx}c{cell_idx}"
            variables[var_name] = move
            domains[var_name] = [self.player]  # Current player's mark is the only option
        
        # 2. Define constraints
        # a. Next move constraints (based on active small board rule)
        if active_small_board_idx != -1:
            constraints["active_board"] = {"type": "active_board", "board_idx": active_small_board_idx}
        
        # b. Win constraints (prioritize winning moves)
        win_constraints = self._find_win_constraints(main_board, small_board_winners, valid_moves)
        if win_constraints:
            constraints["win_opportunities"] = {"type": "win", "opportunities": win_constraints}
        
        # c. Block constraints (block opponent's winning moves)
        block_constraints = self._find_block_constraints(main_board, small_board_winners, valid_moves)
        if block_constraints:
            constraints["block_opportunities"] = {"type": "block", "opportunities": block_constraints}
        
        return {"variables": variables, "domains": domains, "constraints": constraints}
    
    def _find_win_constraints(self, main_board, small_board_winners, valid_moves):
        """Find moves that lead to winning a small board"""
        win_opportunities = []
        
        # Check each valid move to see if it completes a line
        for move in valid_moves:
            board_idx, cell_idx = move
            
            # Skip if board already won
            if small_board_winners[board_idx] != EMPTY:
                continue
            
            # Create temporary board to test move
            temp_board = list(main_board[board_idx])
            temp_board[cell_idx] = self.player
            
            # Check if this move wins the small board
            winner = check_win(temp_board)
            if winner == self.player:
                win_opportunities.append(move)
        
        return win_opportunities
    
    def _find_block_constraints(self, main_board, small_board_winners, valid_moves):
        """Find moves that block opponent from winning a small board"""
        block_opportunities = []
        
        # Check each valid move to see if it blocks opponent
        for move in valid_moves:
            board_idx, cell_idx = move
            
            # Skip if board already won
            if small_board_winners[board_idx] != EMPTY:
                continue
            
            # Create temporary board with opponent's mark to test if they could win
            temp_board = list(main_board[board_idx])
            temp_board[cell_idx] = self.opponent
            
            # Check if opponent would win with this move
            winner = check_win(temp_board)
            if winner == self.opponent:
                block_opportunities.append(move)
        
        return block_opportunities
    
    def _order_moves_by_heuristics(self, csp, valid_moves, main_board, small_board_winners):
        """Order moves using CSP heuristics (MRV, degree, least constraining value)"""
        if not self.use_mrv:
            return valid_moves
        
        # First apply optimization based on objective function (win/block priority)
        if self.use_optimization:
            # Prioritize moves that win or block
            win_moves = csp["constraints"].get("win_opportunities", {}).get("opportunities", [])
            block_moves = csp["constraints"].get("block_opportunities", {}).get("opportunities", [])
            
            # Prioritize order: win > block > other moves
            prioritized = []
            remaining = []
            
            for move in valid_moves:
                if move in win_moves:
                    prioritized.append((move, 3))  # Highest priority
                elif move in block_moves:
                    prioritized.append((move, 2))  # Second priority
                else:
                    remaining.append(move)
            
            # Sort prioritized moves by priority (descending)
            prioritized.sort(key=lambda x: x[1], reverse=True)
            prioritized_moves = [p[0] for p in prioritized]
            
            # Now handle remaining moves with MRV/degree heuristics
            if remaining:
                mrv_ordered = self._apply_mrv_heuristic(remaining, main_board, small_board_winners)
                return prioritized_moves + mrv_ordered
            else:
                return prioritized_moves
        
        # If not using optimization, just use MRV
        return self._apply_mrv_heuristic(valid_moves, main_board, small_board_winners)
    
    def _apply_mrv_heuristic(self, moves, main_board, small_board_winners):
        """Apply Minimum Remaining Values heuristic"""
        board_options = {}
        
        # Count remaining values in each small board
        for board_idx in range(TOTAL_BOARDS):
            if small_board_winners[board_idx] == EMPTY:
                empty_count = main_board[board_idx].count(EMPTY)
                if empty_count > 0:
                    board_options[board_idx] = empty_count
        
        # Calculate a heuristic value for each move
        move_scores = []
        for move in moves:
            board_idx, cell_idx = move
            
            # Base score is remaining values in target board (MRV)
            score = board_options.get(board_idx, float('inf'))
            
            # Apply degree heuristic: prefer moves that lead to boards with fewer options
            target_board_for_opponent = cell_idx
            if 0 <= target_board_for_opponent < TOTAL_BOARDS:
                if small_board_winners[target_board_for_opponent] == EMPTY:
                    # Lower score is better (fewer options means more constrained)
                    degree_score = board_options.get(target_board_for_opponent, float('inf'))
                    if degree_score < float('inf'):
                        # Combine MRV and degree heuristic
                        score = score * 0.5 + degree_score * 0.5
            
            move_scores.append((move, score))
        
        # Sort by score (lower is better - fewer remaining values)
        move_scores.sort(key=lambda x: x[1])
        return [m[0] for m in move_scores]
    
    def _forward_checking(self, move, main_board, small_board_winners):
        """Apply forward checking to reduce domains after move"""
        board_idx, cell_idx = move
        implications = []
        
        # Place the move on the board
        new_board = deep_copy_board(main_board)
        new_board[board_idx][cell_idx] = self.player
        
        # Update small board winners
        new_small_winners = list(small_board_winners)
        small_winner = check_win(new_board[board_idx])
        if small_winner and new_small_winners[board_idx] == EMPTY:
            new_small_winners[board_idx] = small_winner
            implications.append(("win", board_idx, small_winner))
        elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY:
            new_small_winners[board_idx] = DRAW
            implications.append(("draw", board_idx))
        
        # Set next active board
        next_active_board = cell_idx
        
        # Check if next move forces play on a won/full board
        if next_active_board >= 0 and next_active_board < TOTAL_BOARDS:
            if new_small_winners[next_active_board] != EMPTY or is_full(new_board[next_active_board]):
                implications.append(("free_choice", True))
        
        return new_board, new_small_winners, implications
    
    def _run_arc_consistency(self, main_board, small_board_winners, active_small_board_idx):
        """Run AC-3 arc consistency algorithm on UTTT constraints"""
        if not self.use_arc_consistency:
            return [], []
        
        # Initialize queue with all arcs
        queue = SimpleQueue()
        revised_moves = []
        pruned_moves = []
        
        # Get valid moves
        valid_moves = get_valid_moves(main_board, small_board_winners, active_small_board_idx)
        
        # For each valid move, add constraints with other moves
        for move1 in valid_moves:
            board1, cell1 = move1
            
            # Add next-move constraints
            for move2 in valid_moves:
                if move1 != move2:
                    board2, cell2 = move2
                    queue.enqueue((move1, move2))
        
        # Process queue until empty
        while not queue.is_empty():
            move1, move2 = queue.dequeue()
            self.arc_checks += 1
            
            # Check if revising move1 with respect to move2 changes anything
            revised, new_board, new_winners = self._revise(move1, move2, main_board, small_board_winners, active_small_board_idx)
            
            if revised:
                revised_moves.append(move1)
                
                # If a move has been determined superior, prune others
                board1, cell1 = move1
                for move3 in valid_moves:
                    if move3 != move1 and move3 != move2:
                        queue.enqueue((move3, move1))
        
        return revised_moves, pruned_moves
    
    def _revise(self, move1, move2, main_board, small_board_winners, active_small_board_idx):
        """Revise domain of move1 with respect to constraints with move2"""
        revised = False
        new_board = None
        new_winners = None
        
        # Apply move1 and see effect
        board1, cell1 = move1
        temp_board1 = deep_copy_board(main_board)
        temp_board1[board1][cell1] = self.player
        temp_winners1 = list(small_board_winners)
        
        # Check if move1 wins a small board
        small_winner1 = check_win(temp_board1[board1])
        if small_winner1 and temp_winners1[board1] == EMPTY:
            temp_winners1[board1] = small_winner1
            revised = True
        elif is_full(temp_board1[board1]) and temp_winners1[board1] == EMPTY:
            temp_winners1[board1] = DRAW
            revised = True
        
        # Apply move2 and see effect
        board2, cell2 = move2
        temp_board2 = deep_copy_board(main_board)
        temp_board2[board2][cell2] = self.player
        temp_winners2 = list(small_board_winners)
        
        # Check if move2 wins a small board
        small_winner2 = check_win(temp_board2[board2])
        if small_winner2 and temp_winners2[board2] == EMPTY:
            temp_winners2[board2] = small_winner2
            
            # If move2 creates a winning condition, it might constrain move1
            if small_winner2 == self.player:
                revised = True
        
        # If any revision happened, return the new state
        if revised:
            new_board = temp_board1
            new_winners = temp_winners1
        
        return revised, new_board, new_winners
    
    def _csp_search(self, main_board, small_board_winners, current_active_idx, depth, alpha, beta, is_maximizing_player, ordered_moves=None):
        """CSP-based backtracking search with alpha-beta pruning"""
        self.nodes_visited += 1
        
        # Terminal state checks
        game_winner = check_win(small_board_winners)
        all_boards_decided = all(s != EMPTY for s in small_board_winners)
        if game_winner is not None or depth == 0 or all_boards_decided:
            eval_player = self.player
            score = self._evaluate_board(main_board, small_board_winners, eval_player)
            return score, None
        
        current_player_turn = self.player if is_maximizing_player else self.opponent
        
        # If ordered_moves is provided (first call), use them
        # Otherwise, get valid moves for this board state
        valid_moves = ordered_moves if ordered_moves is not None else get_valid_moves(main_board, small_board_winners, current_active_idx)
        
        if not valid_moves:
            eval_player = self.player
            score = self._evaluate_board(main_board, small_board_winners, eval_player)
            return score, None
        
        # Run arc consistency if enabled
        if self.use_arc_consistency and depth == self.depth_limit:
            revised_moves, pruned_moves = self._run_arc_consistency(
                main_board, small_board_winners, current_active_idx)
            
            # If we can prune moves, update valid moves
            if pruned_moves:
                valid_moves = [m for m in valid_moves if m not in pruned_moves]
                if not valid_moves:  # Fallback if we prune everything
                    valid_moves = ordered_moves if ordered_moves is not None else get_valid_moves(main_board, small_board_winners, current_active_idx)
        
        best_move_for_level = valid_moves[0]
        
        if is_maximizing_player:
            max_eval = -math.inf if self.use_alpha_beta else float('-inf')
            for move in valid_moves:
                board_idx, cell_idx = move
                
                # Apply forward checking if enabled
                if self.use_forward_checking:
                    new_board, new_small_winners, implications = self._forward_checking(
                        move, main_board, small_board_winners)
                else:
                    new_board = deep_copy_board(main_board)
                    new_board[board_idx][cell_idx] = current_player_turn
                    new_small_winners = list(small_board_winners)
                    small_winner = check_win(new_board[board_idx])
                    if small_winner and new_small_winners[board_idx] == EMPTY:
                        new_small_winners[board_idx] = small_winner
                    elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY:
                        new_small_winners[board_idx] = DRAW
                
                next_active_board = cell_idx
                
                # Pass None for ordered_moves in recursive calls
                evaluation, _ = self._csp_search(
                    new_board, new_small_winners, next_active_board, 
                    depth - 1, alpha, beta, False, None)
                
                if (self.use_alpha_beta and evaluation > max_eval) or \
                (not self.use_alpha_beta and evaluation > max_eval):
                    max_eval = evaluation
                    best_move_for_level = move
                
                if self.use_alpha_beta:
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break
            
            return max_eval, best_move_for_level
        else:
            min_eval = math.inf if self.use_alpha_beta else float('inf')
            for move in valid_moves:
                board_idx, cell_idx = move
                
                # Apply forward checking if enabled
                if self.use_forward_checking:
                    new_board, new_small_winners, implications = self._forward_checking(
                        move, main_board, small_board_winners)
                    # For opponent's turn, we need to adjust the player
                    new_board[board_idx][cell_idx] = current_player_turn
                else:
                    new_board = deep_copy_board(main_board)
                    new_board[board_idx][cell_idx] = current_player_turn
                    new_small_winners = list(small_board_winners)
                    small_winner = check_win(new_board[board_idx])
                    if small_winner and new_small_winners[board_idx] == EMPTY:
                        new_small_winners[board_idx] = small_winner
                    elif is_full(new_board[board_idx]) and new_small_winners[board_idx] == EMPTY:
                        new_small_winners[board_idx] = DRAW
                
                next_active_board = cell_idx
                
                # Pass None for ordered_moves in recursive calls
                evaluation, _ = self._csp_search(
                    new_board, new_small_winners, next_active_board, 
                    depth - 1, alpha, beta, True, None)
                
                if (self.use_alpha_beta and evaluation < min_eval) or \
                (not self.use_alpha_beta and evaluation < min_eval):
                    min_eval = evaluation
                    best_move_for_level = move
                
                if self.use_alpha_beta:
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break
            
            return min_eval, best_move_for_level

# Add CSP variants to the AI_CLASSES dictionary
AI_CLASSES.update({
    "CSP-FC-AC3-MRV (d=4)": lambda p,o: CSPSolver(p, o, depth_limit=4),
    # "CSP-FC-AC3 (No MRV) (d=4)": lambda p,o: CSPSolver(p, o, use_mrv=False, depth_limit=4),
    # "CSP Basic (No FC/AC3) (d=4)": lambda p,o: CSPSolver(p, o, use_forward_checking=False, use_arc_consistency=False, depth_limit=4),
    # "CSP-FC (No AC3) (d=4)": lambda p,o: CSPSolver(p, o, use_arc_consistency=False, depth_limit=4),
})


class BoardWidget(QWidget):
    def __init__(self, gui_parent):
        super().__init__(gui_parent)
        self.gui = gui_parent 
        self.setMinimumSize(600, 600)
        self.cell_size = 0 
        self.small_board_size = 0

    def calculate_sizes(self):
        widget_size = min(self.width(), self.height())
        effective_size = widget_size 
        self.cell_size = effective_size / (BOARD_SIZE * BOARD_SIZE)
        self.small_board_size = effective_size / BOARD_SIZE
        return effective_size 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_base_size = self.calculate_sizes()

        painter.fillRect(self.rect(), QColor("white"))

        if self.gui.main_board is None or self.gui.small_board_winners is None:
            for i in range(1, BOARD_SIZE * BOARD_SIZE):
                 line_width = 2 if i % BOARD_SIZE == 0 else 1
                 color = QColor("black") if i % BOARD_SIZE == 0 else QColor("lightgray")
                 pen = QPen(color, line_width)
                 pen.setCapStyle(Qt.PenCapStyle.FlatCap) 
                 painter.setPen(pen)
                 x = i * self.cell_size
                 painter.drawLine(int(x), 0, int(x), int(widget_base_size))
                 y = i * self.cell_size
                 painter.drawLine(0, int(y), int(widget_base_size), int(y))
            return 

        thin_pen = QPen(QColor("lightgray"), 1)
        thick_pen = QPen(QColor("black"), 2)
        for i in range(BOARD_SIZE * BOARD_SIZE):
            x = i * self.cell_size
            y = i * self.cell_size
            current_pen = thick_pen if i % BOARD_SIZE == 0 else thin_pen
            painter.setPen(current_pen)
            if i > 0:
                painter.drawLine(int(x), 0, int(x), int(widget_base_size)) 
                painter.drawLine(0, int(y), int(widget_base_size), int(y))
        painter.setPen(thick_pen)
        painter.drawRect(0, 0, int(widget_base_size)-1, int(widget_base_size)-1)


        if not self.gui.game_over:
            valid_moves = get_valid_moves(self.gui.main_board, self.gui.small_board_winners, self.gui.active_small_board_idx)
            active_boards_indices = set(m[0] for m in valid_moves)
            highlight_pen = QPen(QColor(144, 238, 144, 200), 4) 
            highlight_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            painter.setPen(highlight_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for board_idx in active_boards_indices:
                if self.gui.small_board_winners[board_idx] == EMPTY:
                    r_big = board_idx // BOARD_SIZE
                    c_big = board_idx % BOARD_SIZE
                    x0_b = c_big * self.small_board_size
                    y0_b = r_big * self.small_board_size
                    # Adjust rect slightly inwards to avoid overlapping grid lines
                    rect = QRectF(x0_b + 2, y0_b + 2, self.small_board_size - 4, self.small_board_size - 4)
                    painter.drawRect(rect)


        x_pen = QPen(QColor("blue"), 2)
        o_pen = QPen(QColor("red"), 2)
        big_x_pen = QPen(QColor("darkblue"), 5)
        big_o_pen = QPen(QColor("darkred"), 5)
        draw_pen = QPen(QColor("dimgray"), 1) 

        big_symbol_font = QFont("Arial", int(self.small_board_size * 0.5), QFont.Weight.Bold)
        draw_font = QFont("Arial", int(self.small_board_size * 0.6))

        padding_ratio = 0.2 
        big_padding_ratio = 0.1 

        for r_big in range(BOARD_SIZE):
            for c_big in range(BOARD_SIZE):
                board_idx = r_big * BOARD_SIZE + c_big
                small_board = self.gui.main_board[board_idx]
                winner = self.gui.small_board_winners[board_idx]

                if winner != EMPTY:
                    x0_b = c_big * self.small_board_size
                    y0_b = r_big * self.small_board_size
                    board_rect = QRectF(x0_b, y0_b, self.small_board_size, self.small_board_size)

                    if winner == PLAYER_X: fill_color = QColor(208, 208, 255, 180) 
                    elif winner == PLAYER_O: fill_color = QColor(255, 208, 208, 180) 
                    else: fill_color = QColor(224, 224, 224, 180) 
                    painter.fillRect(board_rect, fill_color)

                    pad = self.small_board_size * big_padding_ratio
                    x0_big = x0_b + pad
                    y0_big = y0_b + pad
                    x1_big = x0_b + self.small_board_size - pad
                    y1_big = y0_b + self.small_board_size - pad

                    if winner == PLAYER_X:
                        painter.setPen(big_x_pen)
                        painter.drawLine(int(x0_big), int(y0_big), int(x1_big), int(y1_big))
                        painter.drawLine(int(x0_big), int(y1_big), int(x1_big), int(y0_big))
                    elif winner == PLAYER_O:
                        painter.setPen(big_o_pen)
                        painter.drawEllipse(QRectF(x0_big, y0_big, x1_big - x0_big, y1_big - y0_big))
                    elif winner == DRAW:
                        painter.setPen(draw_pen)
                        painter.setFont(draw_font)
                        painter.drawText(board_rect, Qt.AlignmentFlag.AlignCenter, "D")

                else: 
                    for r_small in range(BOARD_SIZE):
                        for c_small in range(BOARD_SIZE):
                            cell_idx = r_small * BOARD_SIZE + c_small
                            mark = small_board[cell_idx]

                            x0 = (c_big * BOARD_SIZE + c_small) * self.cell_size
                            y0 = (r_big * BOARD_SIZE + r_small) * self.cell_size
                            pad = self.cell_size * padding_ratio
                            x0_c = x0 + pad
                            y0_c = y0 + pad
                            x1_c = x0 + self.cell_size - pad
                            y1_c = y0 + self.cell_size - pad
                            cell_rect = QRectF(x0_c, y0_c, x1_c - x0_c, y1_c - y0_c)

                            if mark == PLAYER_X:
                                painter.setPen(x_pen)
                                painter.drawLine(int(x0_c), int(y0_c), int(x1_c), int(y1_c))
                                painter.drawLine(int(x0_c), int(y1_c), int(x1_c), int(y0_c))
                            elif mark == PLAYER_O:
                                painter.setPen(o_pen)
                                painter.drawEllipse(cell_rect)


    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.gui.is_human_turn():
                return

            widget_base_size = min(self.width(), self.height())
            if self.cell_size <= 0: 
                 self.calculate_sizes()
                 if self.cell_size <= 0: return 

            click_x = event.position().x()
            click_y = event.position().y()

            if not (0 <= click_x < widget_base_size and 0 <= click_y < widget_base_size):
                return

            c = int(click_x // self.cell_size)
            r = int(click_y // self.cell_size)

            if not (0 <= c < BOARD_SIZE*BOARD_SIZE and 0 <= r < BOARD_SIZE*BOARD_SIZE):
                 return

            c_big = c // BOARD_SIZE
            r_big = r // BOARD_SIZE

            c_small = c % BOARD_SIZE
            r_small = r % BOARD_SIZE

            self.gui.handle_board_click(r, c) 


    def sizeHint(self):
        return QSize(600, 600)


# Main Application Window 
class UltimateTicTacToeGUI_Qt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Tic Tac Toe - AI Battles (PyQt6)")
        self.setGeometry(100, 100, 700, 750) 

        self.main_board = None
        self.small_board_winners = None
        self.current_player = None
        self.active_small_board_idx = -1
        self.game_over = True
        self.winner = None
        self.human_player = None
        self.ai = {'X': None, 'O': None}
        self.ai_thinking = False
        self.ai_move_delay = 150 
        self.game_mode = None

        self.total_moves_count = 0
        self.game_stats = {
            'X': {'ai_name': 'Human', 'move_count': 0, 'total_time': 0.0, 'total_nodes': 0},
            'O': {'ai_name': 'Human', 'move_count': 0, 'total_time': 0.0, 'total_nodes': 0}
        }

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.status_label = QLabel("Select Game Mode from Menu")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.status_label.font()
        font.setPointSize(14)
        self.status_label.setFont(font)
        self.main_layout.addWidget(self.status_label)

        self.board_widget = BoardWidget(self) 
        self.main_layout.addWidget(self.board_widget, 1) 

        self.control_widget = QWidget()
        self.control_layout = QGridLayout(self.control_widget) 

        self.p1_label = QLabel("Player X:")
        self.p1_combo = QComboBox()
        self.p1_combo.addItems(AI_CLASSES.keys())

        self.p2_label = QLabel("Player O:")
        self.p2_combo = QComboBox()
        self.p2_combo.addItems(AI_CLASSES.keys())

        self.start_button = QPushButton("Start Game")
        self.start_button.clicked.connect(self.start_selected_game)

        self.control_layout.addWidget(self.p1_label, 0, 0)
        self.control_layout.addWidget(self.p1_combo, 0, 1)
        self.control_layout.addWidget(self.p2_label, 1, 0)
        self.control_layout.addWidget(self.p2_combo, 1, 1)
        self.control_layout.addWidget(self.start_button, 2, 0, 1, 2)

        self.main_layout.addWidget(self.control_widget)
        self.control_widget.setVisible(False)

        self._create_menus()

        self.start_selected_game(initial_setup=True) 

    def _create_menus(self):
        menu_bar = self.menuBar()
        game_menu = menu_bar.addMenu("&Game")

        setup_hva_action = QAction("Setup: &Human vs AI", self)
        setup_hva_action.triggered.connect(lambda: self.setup_mode_selection('hva'))
        game_menu.addAction(setup_hva_action)

        setup_ava_action = QAction("Setup: &AI vs AI", self)
        setup_ava_action.triggered.connect(lambda: self.setup_mode_selection('ava'))
        game_menu.addAction(setup_ava_action)

        game_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close) 
        game_menu.addAction(exit_action)

    def setup_mode_selection(self, mode):
        self.game_mode = mode
        self.control_widget.setVisible(True)

        self.p1_combo.clear()
        self.p1_combo.addItems(AI_CLASSES.keys())
        self.p1_combo.setCurrentIndex(0)
        self.p2_combo.clear()
        self.p2_combo.addItems(AI_CLASSES.keys())
        self.p2_combo.setCurrentIndex(0)

        if mode == 'hva':
            self.status_label.setText("Human vs AI: Choose symbol and AI opponent")
            self.p1_label.setText("Human plays as:")
            self.p1_combo.clear()
            self.p1_combo.addItems(["X", "O"]) 
            self.p2_label.setText("AI Opponent:")
        elif mode == 'ava':
            self.status_label.setText("AI vs AI: Choose AI for X and O")
            self.p1_label.setText("Player X AI:")
            self.p2_label.setText("Player O AI:")


    def start_selected_game(self, initial_setup=False):
         if not initial_setup:
             self.control_widget.setVisible(False)

         self.main_board = [[EMPTY] * TOTAL_BOARDS for _ in range(TOTAL_BOARDS)]
         self.small_board_winners = [EMPTY] * TOTAL_BOARDS
         self.current_player = PLAYER_X
         self.active_small_board_idx = -1
         self.game_over = False
         self.winner = None
         self.ai_thinking = False
         self.ai = {'X': None, 'O': None}
         self.human_player = None
         self.total_moves_count = 0
         self.game_stats = {
             'X': {'ai_name': 'Human', 'move_count': 0, 'total_time': 0.0, 'total_nodes': 0},
             'O': {'ai_name': 'Human', 'move_count': 0, 'total_time': 0.0, 'total_nodes': 0}
         }

         first_ai_to_move = None 

         if not initial_setup and self.game_mode:
             if self.game_mode == 'hva':
                 human_choice = self.p1_combo.currentText() 
                 ai_opponent_type_name = self.p2_combo.currentText()
                 AIClassFactory = AI_CLASSES[ai_opponent_type_name]
                 if human_choice == 'X':
                     self.human_player = PLAYER_X
                     self.ai['O'] = AIClassFactory(PLAYER_O, PLAYER_X)
                     self.game_stats['X']['ai_name'] = 'Human'
                     if self.ai['O']: self.game_stats['O']['ai_name'] = self.ai['O'].name
                     self.status_label.setText(f"Your Turn ({self.human_player})")
                 else:
                     self.human_player = PLAYER_O
                     self.ai['X'] = AIClassFactory(PLAYER_X, PLAYER_O)
                     if self.ai['X']: self.game_stats['X']['ai_name'] = self.ai['X'].name
                     self.game_stats['O']['ai_name'] = 'Human'
                     if self.ai['X']: 
                        self.status_label.setText(f"AI's Turn ({PLAYER_X} - {self.ai['X'].name})")
                        first_ai_to_move = self.ai['X'] 
                     else: 
                        self.status_label.setText("Error creating AI X")
                        self.game_over = True

             elif self.game_mode == 'ava':
                 p1_ai_type_name = self.p1_combo.currentText()
                 p2_ai_type_name = self.p2_combo.currentText()
                 AIClassXFactory = AI_CLASSES[p1_ai_type_name]
                 AIClassOFactory = AI_CLASSES[p2_ai_type_name]
                 self.ai['X'] = AIClassXFactory(PLAYER_X, PLAYER_O)
                 self.ai['O'] = AIClassOFactory(PLAYER_O, PLAYER_X)
                 if self.ai['X']: self.game_stats['X']['ai_name'] = self.ai['X'].name
                 if self.ai['O']: self.game_stats['O']['ai_name'] = self.ai['O'].name

                 if self.ai['X']: 
                     self.status_label.setText(f"AI X's Turn ({self.ai['X'].name})")
                     first_ai_to_move = self.ai['X']
                 else: 
                     self.status_label.setText("Error creating AI X")
                     self.game_over = True
         else:
              self.status_label.setText("Select Game Mode from Menu")
              self.game_over = True 

         self.board_widget.update() 

         if first_ai_to_move:
             QTimer.singleShot(100, self.trigger_ai_move) 

    def is_human_turn(self):
        return not self.game_over and not self.ai_thinking and self.human_player == self.current_player

    @pyqtSlot(int, int) 
    def handle_board_click(self, r, c):
        c_big = c // BOARD_SIZE
        r_big = r // BOARD_SIZE
        board_idx = r_big * BOARD_SIZE + c_big

        c_small = c % BOARD_SIZE
        r_small = r % BOARD_SIZE
        cell_idx = r_small * BOARD_SIZE + c_small

        move = (board_idx, cell_idx)

        valid_moves = get_valid_moves(self.main_board, self.small_board_winners, self.active_small_board_idx)

        if move in valid_moves:
            self.make_move(board_idx, cell_idx)
            if not self.game_over and self.ai[self.current_player]:
                 QTimer.singleShot(100, self.trigger_ai_move)

    def make_move(self, board_idx, cell_idx):
        if self.game_over or self.main_board[board_idx][cell_idx] != EMPTY: return

        player_who_moved = self.current_player
        self.main_board[board_idx][cell_idx] = player_who_moved

        # Update move counts
        self.total_moves_count += 1
        if player_who_moved in self.game_stats:
            self.game_stats[player_who_moved]['move_count'] += 1

        # Check small board win/draw
        small_winner = check_win(self.main_board[board_idx])
        if small_winner and self.small_board_winners[board_idx] == EMPTY:
            self.small_board_winners[board_idx] = small_winner
        elif is_full(self.main_board[board_idx]) and self.small_board_winners[board_idx] == EMPTY:
            self.small_board_winners[board_idx] = DRAW

        # Check overall game win/draw
        self.winner = check_win(self.small_board_winners) # Check win on the big board
        game_ended_this_turn = False
        end_message = ""
        if self.winner:
            self.game_over = True; game_ended_this_turn = True
            end_message = f"Player {self.winner} wins!"
            self.status_label.setText(f"Game Over! Player {self.winner} wins!")
        elif all(s != EMPTY for s in self.small_board_winners):
            self.game_over = True; game_ended_this_turn = True
            self.winner = DRAW
            end_message = "It's a Draw!"
            self.status_label.setText("Game Over! It's a Draw!")

        if game_ended_this_turn:
             self.board_widget.update()
             QMessageBox.information(self, "Game Over", end_message)
             self.print_game_summary() 
             return 

        self.active_small_board_idx = cell_idx
        self.current_player = PLAYER_O if player_who_moved == PLAYER_X else PLAYER_X
        if self.human_player == self.current_player:
            self.status_label.setText(f"Your Turn ({self.human_player})")
        elif self.ai[self.current_player]:
            ai_name = self.ai[self.current_player].name
            self.status_label.setText(f"AI {self.current_player}'s Turn ({ai_name})")

        self.board_widget.update()


    def trigger_ai_move(self):
        if self.game_over or self.ai_thinking: return
        current_ai = self.ai.get(self.current_player) 
        if current_ai:
            self.ai_thinking = True
            self.status_label.setText(f"AI {self.current_player} ({current_ai.name}) is thinking...")
            QTimer.singleShot(50, self._execute_ai_move) 


    @pyqtSlot()
    def _execute_ai_move(self):
        if self.game_over: self.ai_thinking = False; return

        player_symbol = self.current_player 
        current_ai = self.ai.get(player_symbol)
        if not current_ai: self.ai_thinking = False; return

        board_copy = deep_copy_board(self.main_board)
        winners_copy = list(self.small_board_winners)
        active_idx_copy = self.active_small_board_idx

        ai_result = current_ai.get_move(board_copy, winners_copy, active_idx_copy)
        if ai_result is None: move, time_taken, nodes_visited = None, 0.0, 0
        else: move, time_taken, nodes_visited = ai_result

        if player_symbol in self.game_stats:
            self.game_stats[player_symbol]['total_time'] += time_taken
            self.game_stats[player_symbol]['total_nodes'] += nodes_visited if isinstance(nodes_visited, (int, float)) else 0

        if move: print(f" AI {player_symbol} ({current_ai.name}): Move {move}, Time: {time_taken:.3f}s, Nodes: {nodes_visited:,}")
        else: print(f" AI {player_symbol} ({current_ai.name}) returned no move. Time: {time_taken:.3f}s, Nodes: {nodes_visited:,}")

        self.ai_thinking = False 

        if self.game_over: return

        if move:
            valid_moves = get_valid_moves(self.main_board, self.small_board_winners, self.active_small_board_idx)
            if move in valid_moves:
                 self.make_move(move[0], move[1]) 
                 if not self.game_over and not self.human_player and self.ai.get(self.current_player):
                     QTimer.singleShot(self.ai_move_delay, self.trigger_ai_move)
            else:
                 print(f"CRITICAL ERROR: AI {current_ai.name} ({player_symbol}) returned invalid move {move}. Valid: {valid_moves}")
                 if valid_moves:
                     fallback_move = random.choice(valid_moves); print("Picking random valid move as fallback.")
                     self.make_move(fallback_move[0], fallback_move[1])
                     if not self.game_over and not self.human_player and self.ai.get(self.current_player): QTimer.singleShot(self.ai_move_delay, self.trigger_ai_move)
                 else:
                      self.status_label.setText("Error: AI failed, no valid moves?"); self.game_over = True
                      if not self.winner: self.winner = DRAW
                      self.board_widget.update() 
                      self.print_game_summary()

        else: 
            if not get_valid_moves(self.main_board, self.small_board_winners, self.active_small_board_idx):
                 if not self.game_over:
                      print("No valid moves left. Declaring Draw.")
                      self.winner = DRAW; self.game_over = True
                      self.status_label.setText("Game Over! It's a Draw! (No moves left)")
                      self.board_widget.update() 
                      QMessageBox.information(self, "Game Over", "It's a Draw!")
                      self.print_game_summary()
            else:
                print(f"CRITICAL ERROR: AI {current_ai.name} ({player_symbol}) returned None, but valid moves exist!")
                valid_moves = get_valid_moves(self.main_board, self.small_board_winners, self.active_small_board_idx)
                if valid_moves:
                     fallback_move = random.choice(valid_moves); print("Picking random valid move as fallback.")
                     self.make_move(fallback_move[0], fallback_move[1])
                     if not self.game_over and not self.human_player and self.ai.get(self.current_player): QTimer.singleShot(self.ai_move_delay, self.trigger_ai_move)
                else:
                     self.status_label.setText("Error: AI failed, inconsistent move state?"); self.game_over = True
                     if not self.winner: self.winner = DRAW
                     self.board_widget.update()
                     self.print_game_summary()


    def print_game_summary(self):
        summary = f"\n================== GAME SUMMARY ==================\n"
        if self.winner == DRAW: summary += " Result: DRAW\n"
        else: summary += f" Result: Player {self.winner} Wins!\n"
        summary += f" Total Moves: {self.total_moves_count}\n"
        x_wins = self.small_board_winners.count(PLAYER_X)
        o_wins = self.small_board_winners.count(PLAYER_O)
        draws = self.small_board_winners.count(DRAW)
        summary += f" Small Boards Won (X): {x_wins}\n"
        summary += f" Small Boards Won (O): {o_wins}\n"
        summary += f" Small Boards Drawn: {draws}\n"

        summary += "\n --- Player X Stats ---\n"
        stats_x = self.game_stats.get('X', {})
        ai_name_x = stats_x.get('ai_name', 'N/A'); move_count_x = stats_x.get('move_count', 0)
        total_time_x = stats_x.get('total_time', 0.0); total_nodes_x = stats_x.get('total_nodes', 0)
        summary += f"  Type: {ai_name_x}\n"; summary += f"  Moves: {move_count_x}\n"
        if ai_name_x != 'Human':
            avg_time_x = total_time_x / move_count_x if move_count_x > 0 else 0
            avg_nodes_x = total_nodes_x / move_count_x if move_count_x > 0 else 0
            summary += f"  Total Time: {total_time_x:.3f}s\n"; summary += f"  Avg Time/Move: {avg_time_x:.3f}s\n"
            if total_nodes_x > 0: summary += f"  Total Nodes: {total_nodes_x:,}\n"; summary += f"  Avg Nodes/Move: {avg_nodes_x:,.1f}\n"

        summary += "\n --- Player O Stats ---\n"
        stats_o = self.game_stats.get('O', {})
        ai_name_o = stats_o.get('ai_name', 'N/A'); move_count_o = stats_o.get('move_count', 0)
        total_time_o = stats_o.get('total_time', 0.0); total_nodes_o = stats_o.get('total_nodes', 0)
        summary += f"  Type: {ai_name_o}\n"; summary += f"  Moves: {move_count_o}\n"
        if ai_name_o != 'Human':
            avg_time_o = total_time_o / move_count_o if move_count_o > 0 else 0
            avg_nodes_o = total_nodes_o / move_count_o if move_count_o > 0 else 0
            summary += f"  Total Time: {total_time_o:.3f}s\n"; summary += f"  Avg Time/Move: {avg_time_o:.3f}s\n"
            if total_nodes_o > 0: summary += f"  Total Nodes: {total_nodes_o:,}\n"; summary += f"  Avg Nodes/Move: {avg_nodes_o:,.1f}\n"

        summary += "=================================================="
        print(summary) 


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = UltimateTicTacToeGUI_Qt() 
    main_window.show()
    sys.exit(app.exec())
