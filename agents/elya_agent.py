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
        """
        Decide the next move based on the game phase and adapt to board size.
        """
        # Ensure board validity
        if board is None or not isinstance(board, np.ndarray):
            raise ValueError("Invalid board passed to the agent.")
        
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        # Determine the game phase based on the percentage of empty cells
        empty_cells = np.sum(board == 0)
        total_cells = board.shape[0] * board.shape[1]
        empty_ratio = empty_cells / total_cells

        if empty_ratio > 0.6:  # Opening phase
            return self.opening_phase(board, color, legal_moves)
        elif empty_ratio > 0.2:  # Midgame phase
            return self.midgame_phase(board, color, opponent, legal_moves, board.shape[0])
        else:  # Endgame phase
            return self.endgame_phase(board, color, legal_moves)

    def opening_phase(self, board, color, legal_moves):
        """
        Opening phase strategy: prioritize corners and maximize mobility.
        """
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]

        # Prioritize corners if available
        for move in legal_moves:
            if move in corners:
                return move

        # Define mobility function to avoid AttributeError
        def mobility(move):
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            return len(get_valid_moves(simulated_board, color))

        # Otherwise, maximize mobility
        return max(legal_moves, key=mobility)

    def midgame_phase(self, board, color, opponent, legal_moves, board_size):
        """
        Midgame phase strategy: balance corners, stability, and limiting opponent mobility.
        """
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            move_score = self.evaluate_board(simulated_board, color, opponent, board_size)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        return best_move if best_move else random.choice(legal_moves)

    def endgame_phase(self, board, color, legal_moves):
        """
        Endgame phase strategy: prioritize capturing the most discs and ensuring stability.
        """
        return max(legal_moves, key=lambda move: count_capture(board, move, color))

    def evaluate_board(self, board, color, opponent, board_size):
        """
        Evaluate the board state dynamically based on board size.
        """
        corners = [(0, 0), (0, board.shape[1] - 1),
                   (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

        # Adjust weights dynamically for large boards
        corner_weight = 30 if board_size <= 8 else 50
        mobility_weight = -3 if board_size <= 8 else -5
        stability_weight = 5 if board_size <= 8 else 10

        # Corner positions are highly valuable
        corner_score = sum(corner_weight if board[x, y] == color else -corner_weight if board[x, y] == opponent else 0
                           for x, y in corners)

        # Penalize moves adjacent to unoccupied corners
        adjacent_penalty = self.adjacent_to_corners_penalty(board, color)

        # Stability: stable discs that cannot be flipped
        stability_score = self.count_stable_discs(board, color) * stability_weight

        # Limit opponent mobility
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_score = opponent_moves * mobility_weight

        return corner_score + adjacent_penalty + stability_score + mobility_score

    def adjacent_to_corners_penalty(self, board, color):
        """
        Penalize moves adjacent to unoccupied corners.
        """
        adjacent_to_corners = [
            (0, 1), (1, 0), (1, 1),
            (0, board.shape[1] - 2), (1, board.shape[1] - 1), (1, board.shape[1] - 2),
            (board.shape[0] - 1, 1), (board.shape[0] - 2, 0), (board.shape[0] - 2, 1),
            (board.shape[0] - 1, board.shape[1] - 2),
            (board.shape[0] - 2, board.shape[1] - 1),
            (board.shape[0] - 2, board.shape[1] - 2)
        ]
        return sum(-10 for adj in adjacent_to_corners if board[adj] == color)

    def count_stable_discs(self, board, color):
        """
        Count the number of stable discs for the given player.
        """
        stable_count = 0
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r, c] == color and self.is_stable(board, (r, c), color):
                    stable_count += 1
        return stable_count

    def is_stable(self, board, position, color):
        """
        Determine if a disc at the given position is stable.
        """
        x, y = position
        rows, cols = board.shape

        # Discs in the corners or edges are likely stable
        if x in {0, rows - 1} or y in {0, cols - 1}:
            return True

        return False


