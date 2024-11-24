from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, count_capture, execute_move, check_endgame, get_directions
import sys
import numpy as np
from copy import deepcopy
import time

class TimeoutException(Exception):
    pass

@register_agent("opp1")
class Opp1Agent(Agent):
    """
    A custom agent for playing Reversi/Othello using Alpha-Beta Pruning with depth-limited iterative deepening.
    """

    def __init__(self):
        super(Opp1Agent, self).__init__()
        self.name = "opp1"
        self.time_limit = 1.99  # Max 2 seconds per turn for the search

    def step(self, chess_board, player, opponent):
        """
        Determine the next move for the agent.
        """
        start_time = time.time()
        i = 1
        best_move = None
        while True:
            try:
                temp = self.alpha_beta_pruning_depth_limited(
                    chess_board, player, opponent, start_time, i
                )
                if temp:
                    best_move = temp
                else:
                    break
                i += 1
            except TimeoutException:
                break

        time_taken = time.time() - start_time
        print(f"My AI's turn took {time_taken} seconds.")
        return best_move

    def evaluate(self, chess_board, player, opponent) -> float:
        """
        Evaluate the board state for the given player.
        """

        def count_stable_discs(chess_board, player):
            stable_discs = 0
            directions = get_directions()
            rows, cols = chess_board.shape
            for r in range(rows):
                for c in range(cols):
                    if chess_board[r, c] == player:
                        stable = True
                        for dr, dc in directions:
                            rr, cc = r, c
                            while 0 <= rr < rows and 0 <= cc < cols:
                                if chess_board[rr, cc] == 0:
                                    stable = False
                                    break
                                rr += dr
                                cc += dc
                            if not stable:
                                break
                        if stable:
                            stable_discs += 1
            return stable_discs

        def edge_occupancy(chess_board, player):
            rows, cols = chess_board.shape
            return (
                np.sum(chess_board[0] == player)
                + np.sum(chess_board[rows - 1] == player)
                + np.sum(chess_board[:, 0] == player)
                + np.sum(chess_board[:, cols - 1] == player)
            )

        player_pieces = np.sum(chess_board == player)
        opponent_pieces = np.sum(chess_board == opponent)
        piece_advantage = player_pieces - opponent_pieces

        player_mobility = len(get_valid_moves(chess_board, player))
        opponent_mobility = len(get_valid_moves(chess_board, opponent))
        mobility_advantage = player_mobility - opponent_mobility

        player_stability = count_stable_discs(chess_board, player)
        opponent_stability = count_stable_discs(chess_board, opponent)
        stability_advantage = player_stability - opponent_stability

        player_edge_occupancy = edge_occupancy(chess_board, player)
        opponent_edge_occupancy = edge_occupancy(chess_board, opponent)
        edge_occupancy_advantage = player_edge_occupancy - opponent_edge_occupancy

        eval = (
            piece_advantage * 1.0
            + mobility_advantage * 1.0
            + stability_advantage * 3.0
            + edge_occupancy_advantage * 4.0
        )
        return eval

    def alpha_beta_pruning_depth_limited(
        self, chess_board: np.array, player, opponent, start_time: float, depth_limit: int
    ):

        def max_value(board, alpha, beta, depth) -> float:
            if time.time() - start_time > self.time_limit:
                raise TimeoutException()
            best_move = None
            if depth == depth_limit or check_endgame(board, player, opponent)[0]:
                return self.evaluate(board, player, opponent)

            valid_moves = get_valid_moves(chess_board, player)
            if not valid_moves:
                return min_value(chess_board, alpha, beta, depth)

            for move in valid_moves:
                board_copy = board.copy()
                execute_move(board_copy, move, player)
                min_val = min_value(board_copy, alpha, beta, depth + 1)
                if alpha < min_val:
                    alpha = min_val
                    if depth == 0:
                        best_move = move

                if alpha >= beta:
                    return beta
            return best_move if depth == 0 else alpha

        def min_value(board, alpha, beta, depth) -> float:
            if time.time() - start_time > self.time_limit:
                raise TimeoutException()
            if depth == depth_limit or check_endgame(board, opponent, player):
                return self.evaluate(board, player, opponent)

            valid_moves = get_valid_moves(chess_board, opponent)
            if not valid_moves:
                return max_value(chess_board, alpha, beta, depth)

            for move in valid_moves:
                board_copy = board.copy()
                execute_move(board_copy, move, opponent)
                max_val = max_value(board_copy, alpha, beta, depth + 1)
                beta = min(beta, max_val)

                if alpha >= beta:
                    return alpha
            return beta

        try:
            INF = sys.maxsize
            return max_value(chess_board, -INF, +INF, 0)
        except TimeoutException:
            return None

  

