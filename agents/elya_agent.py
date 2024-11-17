from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, count_capture, execute_move, check_endgame
import copy
import random
import numpy as np


@register_agent("elya_agent")
class ElyaAgent(Agent):

    def __init__(self):
        super().__init__()
        self.name = "elya_agent"

    def step(self, board, color, opponent):
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        # Determine the game phase based on the number of empty cells
        empty_cells = np.sum(board == 0)
        if empty_cells > 40:  # Opening phase
            return self.opening_phase(board, color, legal_moves)
        elif empty_cells > 10:  # Midgame phase
            return self.midgame_phase(board, color, opponent, legal_moves)
        else:  # Endgame phase
            return self.endgame_phase(board, color, legal_moves)

    def opening_phase(self, board, color, legal_moves):
        """
        Opening phase strategy: prioritize corners and maximize mobility.
        """
        # Define corner positions
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]

        # Prioritize corners if available
        for move in legal_moves:
            if move in corners:
                return move

        # Otherwise, choose a move that maximizes mobility
        return max(legal_moves, key=lambda move: len(get_valid_moves(execute_move(copy.deepcopy(board), move, color), color)))

    def midgame_phase(self, board, color, opponent, legal_moves):
        """
        Midgame phase strategy: prioritize corners, minimize opponent mobility, and evaluate stability.
        """
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            move_score = self.evaluate_board(simulated_board, color, opponent)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        return best_move if best_move else random.choice(legal_moves)

    def endgame_phase(self, board, color, legal_moves):
        """
        Endgame phase strategy: maximize immediate captures and stable discs.
        """
        return max(legal_moves, key=lambda move: count_capture(board, move, color))

    def evaluate_board(self, board, color, opponent):
        corners = [(0, 0), (0, board.shape[1] - 1),
                   (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

        # Corner positions are highly valuable
        corner_score = sum(1 for corner in corners if board[corner] == color) * 30
        corner_penalty = sum(1 for corner in corners if board[corner] == opponent) * -30

        # Penalize moves adjacent to corners if the corner is unoccupied
        adjacent_to_corners = [
            (0, 1), (1, 0), (1, 1), 
            (0, board.shape[1] - 2), (1, board.shape[1] - 1), (1, board.shape[1] - 2),
            (board.shape[0] - 1, 1), (board.shape[0] - 2, 0), (board.shape[0] - 2, 1),
            (board.shape[0] - 1, board.shape[1] - 2), 
            (board.shape[0] - 2, board.shape[1] - 1), 
            (board.shape[0] - 2, board.shape[1] - 2)
        ]
        adjacent_penalty = sum(-10 for adj in adjacent_to_corners if board[adj] == color)

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_score = -opponent_moves * 5

        # Stability: stable discs cannot be flipped
        stability_score = self.count_stable_discs(board, color) * 5

        # Combine scores
        return corner_score + corner_penalty + adjacent_penalty + mobility_score + stability_score

    def count_stable_discs(self, board, color):
        stable_count = 0
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r, c] == color and self.is_stable(board, (r, c), color):
                    stable_count += 1
        return stable_count

    def is_stable(self, board, position, color):
        x, y = position
        rows, cols = board.shape

        # Discs in the corners or edges are likely stable
        if x in {0, rows - 1} or y in {0, cols - 1}:
            return True

        # Add further stability checks for more precision
        return False

