from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, count_capture, execute_move, check_endgame, get_directions
import sys
import numpy as np
import time


@register_agent("opp2_agent")
class Opp2Agent(Agent):
    """
    A custom agent for playing Reversi/Othello using Alpha-Beta Pruning with advanced heuristics.
    """

    def __init__(self):
        super(Opp2Agent, self).__init__()
        self.name = "opp2_agent"

    def step(self, chess_board, player, opponent):
        """
        Decide the next move using Alpha-Beta pruning with iterative deepening.
        """
        start_time = time.time()
        num_pieces = np.count_nonzero(chess_board)
        num_moves = len(get_valid_moves(chess_board, player))

        best_move = None
        depth = 1
        while True:
            try:
                temp_move = self.alpha_beta_pruning(chess_board, player, opponent, start_time, depth)
                if temp_move:
                    best_move = temp_move
                depth += 1
            except TimeoutError:
                break

        time_taken = time.time() - start_time
        print(f"My AI's turn took {time_taken} seconds.")
        return best_move

    def alpha_beta_pruning(self, chess_board, player, opponent, start_time, depth_limit):
        """
        Alpha-Beta Pruning with depth limitation.
        """

        def max_value(board, alpha, beta, depth):
            if time.time() - start_time > 1.99:
                raise TimeoutError
            if depth == depth_limit or check_endgame(board, player, opponent)[0]:
                return self.evaluate_board(board, player, opponent)

            valid_moves = get_valid_moves(board, player)
            if not valid_moves:
                return min_value(board, alpha, beta, depth)

            max_eval = float('-inf')
            for move in valid_moves:
                new_board = board.copy()
                execute_move(new_board, move, player)
                eval_score = min_value(new_board, alpha, beta, depth + 1)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if alpha >= beta:
                    break
            return max_eval

        def min_value(board, alpha, beta, depth):
            if time.time() - start_time > 1.99:
                raise TimeoutError
            if depth == depth_limit or check_endgame(board, opponent, player)[0]:
                return self.evaluate_board(board, player, opponent)

            valid_moves = get_valid_moves(board, opponent)
            if not valid_moves:
                return max_value(board, alpha, beta, depth)

            min_eval = float('inf')
            for move in valid_moves:
                new_board = board.copy()
                execute_move(new_board, move, opponent)
                eval_score = max_value(new_board, alpha, beta, depth + 1)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if alpha >= beta:
                    break
            return min_eval

        valid_moves = get_valid_moves(chess_board, player)
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in valid_moves:
            new_board = chess_board.copy()
            execute_move(new_board, move, player)
            eval_score = min_value(new_board, alpha, beta, 1)
            if eval_score > best_score:
                best_score = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break

        return best_move

    def evaluate_board(self, board, player, opponent):
        """
        Evaluate the board state using weighted positional evaluation and other factors.
        """
        POSITIONAL_WEIGHTS = self.get_positional_weights(len(board))
        piece_advantage = np.sum(board == player) - np.sum(board == opponent)
        mobility_advantage = len(get_valid_moves(board, player)) - len(get_valid_moves(board, opponent))
        position_value = self.calculate_position_value(board, player, opponent, POSITIONAL_WEIGHTS)
        return piece_advantage + mobility_advantage + position_value

    def calculate_position_value(self, board, player, opponent, weights):
        """
        Calculate the positional value of the board for the given player.
        """
        value = 0
        for r in range(len(board)):
            for c in range(len(board)):
                if board[r, c] == player:
                    value += weights[r][c]
                elif board[r, c] == opponent:
                    value -= weights[r][c]
        return value

    def get_positional_weights(self, size):
        """
        Return positional weights for different board sizes.
        """
        weights_6x6 = [
            [50, -20, -10, -10, -20, 50],
            [-20, -50, -2, -2, -50, -20],
            [-10, -2, 5, 5, -2, -10],
            [-10, -2, 5, 5, -2, -10],
            [-20, -50, -2, -2, -50, -20],
            [50, -20, -10, -10, -20, 50]
        ]
        weights_8x8 = [
            [50, -20, 10, 5, 5, 10, -20, 50],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, 5, 1, 1, 5, -2, 10],
            [5, -2, 1, 1, 1, 1, -2, 5],
            [5, -2, 1, 1, 1, 1, -2, 5],
            [10, -2, 5, 1, 1, 5, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [50, -20, 10, 5, 5, 10, -20, 50]
        ]
        weights_10x10 = [
            [50, -20, -10, 5, 5, 5, 5, -10, -20, 50],
            [-20, -50, -2, -2, -2, -2, -2, -2, -50, -20],
            [-10, -2, 5, 1, 1, 1, 1, 5, -2, -10],
            [5, -2, 1, 1, 1, 1, 1, 1, -2, 5],
            [5, -2, 1, 1, 1, 1, 1, 1, -2, 5],
            [5, -2, 1, 1, 1, 1, 1, 1, -2, 5],
            [-10, -2, 5, 1, 1, 1, 1, 5, -2, -10],
            [-20, -50, -2, -2, -2, -2, -2, -2, -50, -20],
            [50, -20, -10, 5, 5, 5, 5, -10, -20, 50]
        ]
        if size == 6:
            return weights_6x6
        elif size == 8:
            return weights_8x8
        elif size == 10:
            return weights_10x10
        else:
            raise ValueError("Unsupported board size")
