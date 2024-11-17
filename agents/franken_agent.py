from agents.agent import Agent
from store import register_agent
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import numpy as np
from copy import deepcopy
import time

@register_agent("franken_agent")
class FrankenAgent(Agent):

    def __init__(self):
        super(FrankenAgent, self).__init__()
        self.name = "FrankenAgent"

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        valid_moves = get_valid_moves(chess_board, player)

        if not valid_moves:
            return None  # No valid moves available, pass turn

        # Determine the phase of the game
        empty_cells = np.sum(chess_board == 0)
        if empty_cells > 40:  # Opening phase
            move = self.opening_phase(chess_board, player, valid_moves)
        elif empty_cells > 10:  # Midgame phase with simulated annealing
            move = self.midgame_with_simulated_annealing(chess_board, player, opponent, valid_moves)
        else:  # Endgame phase
            move = self.endgame_phase(chess_board, player, valid_moves)

        time_taken = time.time() - start_time
        print(f"FrankenAgent's turn took {time_taken:.4f} seconds.")
        return move

    def opening_phase(self, board, player, legal_moves):
        """
        Opening phase strategy: prioritize corners and maximize mobility.

        Parameters:
        - board: 2D numpy array representing the game board.
        - player: Integer representing the agent's color.
        - legal_moves: List of valid moves.

        Returns:
        - Tuple (x, y): The chosen move.
        """
        # Define corner positions
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]

        # Prioritize corners if available
        for move in legal_moves:
            if move in corners:
                return move

        # Otherwise, choose a move that maximizes mobility
        return max(legal_moves, key=lambda move: len(get_valid_moves(execute_move(deepcopy(board), move, player), player)))

    def midgame_with_simulated_annealing(self, board, player, opponent, legal_moves):
        # Simulated annealing parameters
        initial_temp = 100
        alpha = 0.95
        min_temp = 1

        def heuristic(board, player):
            """
            Heuristic function to evaluate the board state.
            Higher scores indicate better states for the player.
            """
            # Number of discs owned
            player_discs = np.sum(board == player)
            opponent_discs = np.sum(board == (3 - player))

            # Corner positions are advantageous
            corners = [(0, 0), (0, board.shape[1] - 1),
                       (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
            corner_score = sum(5 if board[x, y] == player else -5 if board[x, y] == (3 - player) else 0 for x, y in corners)

            # Total heuristic score
            return player_discs - opponent_discs + corner_score

        current_board = deepcopy(board)
        current_temp = initial_temp
        best_move = None

        while current_temp > min_temp:
            if not legal_moves:
                break

            # Randomly choose a neighbor state (valid move)
            move = legal_moves[np.random.randint(len(legal_moves))]
            new_board = deepcopy(current_board)
            execute_move(new_board, move, player)

            # Calculate heuristic changes
            current_heuristic = heuristic(current_board, player)
            new_heuristic = heuristic(new_board, player)
            delta_e = new_heuristic - current_heuristic

            # Acceptance probability
            if delta_e > 0 or np.random.rand() < np.exp(delta_e / current_temp):
                current_board = new_board
                best_move = move

            # Cool down the temperature
            current_temp *= alpha

        return best_move if best_move else random_move(board, player)

    def endgame_phase(self, board, player, legal_moves):
        # Maximize disc captures in the endgame
        return max(legal_moves, key=lambda move: count_capture(board, move, player))
