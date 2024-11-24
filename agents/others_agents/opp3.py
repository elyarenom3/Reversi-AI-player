from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("opp3")
class Opp3Agent(Agent):
    """
    A custom agent for playing Reversi/Othello with advanced heuristics.
    """

    def __init__(self):
        super(Opp3Agent, self).__init__()
        self.name = "opp3"

        # Potential weights for each heuristic component
        self.weights = {
            "coin_parity": 1.0,
            "mobility": 2.0,
            "corners_captured": 9.0,
            "stability": 8.0
        }

    def heuristic_coin_parity(self, chess_board, player, opponent):
        player_coins = np.sum(chess_board == player)
        opponent_coins = np.sum(chess_board == opponent)
        if player_coins + opponent_coins != 0:
            return (player_coins - opponent_coins) / (player_coins + opponent_coins)
        return 0

    def heuristic_mobility(self, chess_board, player, opponent):
        player_mobility = len(get_valid_moves(chess_board, player))
        opponent_mobility = len(get_valid_moves(chess_board, opponent))
        if player_mobility + opponent_mobility != 0:
            return (player_mobility - opponent_mobility) / (player_mobility + opponent_mobility)
        return 0

    def heuristic_corners_capture(self, chess_board, player, opponent):
        corners = [
            (0, 0),
            (0, len(chess_board) - 1),
            (len(chess_board) - 1, 0),
            (len(chess_board) - 1, len(chess_board) - 1)
        ]
        player_corners = sum(1 for x, y in corners if chess_board[x][y] == player)
        opponent_corners = sum(1 for x, y in corners if chess_board[x][y] == opponent)
        if player_corners + opponent_corners != 0:
            return (player_corners - opponent_corners) / (player_corners + opponent_corners)
        return 0

    def heuristic_stability(self, chess_board, player, opponent):
        board_size = len(chess_board)
        player_stability = 0
        opponent_stability = 0

        for r in range(board_size):
            for c in range(board_size):
                if chess_board[r][c] == 0:  # Board space not occupied
                    continue
                elif chess_board[r][c] == player:
                    player_stability += self.classify_stability(chess_board, r, c, player, opponent)
                elif chess_board[r][c] == opponent:
                    opponent_stability += self.classify_stability(chess_board, r, c, opponent, player)

        if player_stability + opponent_stability != 0:
            return (player_stability - opponent_stability) / (player_stability + opponent_stability)
        return 0

    def classify_stability(self, chess_board, r, c, player, opponent):
        """
        Classify the stability of a disc at position (r, c).
        """
        board_size = len(chess_board)
        if (r, c) in [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]:
            return 1
        if self.is_unstable(chess_board, r, c, opponent):
            return -1
        if self.is_stable(chess_board, r, c, player):
            return 1
        return 0

    def is_unstable(self, chess_board, r, c, opponent):
        """
        Determine if a disc can be flanked in the next move.
        """
        board_size = len(chess_board)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            x, y = r + dr, c + dc
            if 0 <= x < board_size and 0 <= y < board_size:
                if chess_board[x][y] == opponent:
                    nx, ny = x + dr, y + dc
                    while 0 <= nx < board_size and 0 <= ny < board_size:
                        if chess_board[nx][ny] == 0:
                            break
                        if chess_board[nx][ny] == chess_board[r][c]:
                            return True
                        nx += dr
                        ny += dc
        return False

    def is_stable(self, chess_board, r, c, player):
        """
        Check if a disc is stable.
        """
        board_size = len(chess_board)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                if chess_board[nr][nc] != player and chess_board[nr][nc] != 0:
                    return False
        return True

    def evaluation_function(self, chess_board, player, opponent):
        """
        Combine all heuristic components with weights.
        """
        coin_parity_score = self.heuristic_coin_parity(chess_board, player, opponent)
        mobility_score = self.heuristic_mobility(chess_board, player, opponent)
        corners_score = self.heuristic_corners_capture(chess_board, player, opponent)
        stability_score = self.heuristic_stability(chess_board, player, opponent)

        total_score = (
            self.weights["coin_parity"] * coin_parity_score +
            self.weights["mobility"] * mobility_score +
            self.weights["corners_captured"] * corners_score +
            self.weights["stability"] * stability_score
        )
        return total_score

    def step(self, chess_board, player, opponent):
        """
        Decide the next move based on the evaluation function.
        """
        legal_moves = get_valid_moves(chess_board, player)
        if not legal_moves:
            return None

        best_move = None
        best_score = -float("inf")
        for move in legal_moves:
            simulated_board = deepcopy(chess_board)
            execute_move(simulated_board, move, player)
            score = self.evaluation_function(simulated_board, player, opponent)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
