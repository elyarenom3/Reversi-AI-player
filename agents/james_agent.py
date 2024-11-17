from agents.agent import Agent
from store import register_agent
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import sys
import numpy as np
from copy import deepcopy
import time

@register_agent("james_agent")
class JamesAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(JamesAgent, self).__init__()
        self.name = "JamesAgent"

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
          where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
          and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        Returns:
        - Tuple (r, c): The position (row, col) where the agent places its next disc.
        """

        start_time = time.time()

        # Simulated annealing parameters
        initial_temp = 100
        alpha = 0.95
        min_temp = 1

        # Simulated annealing logic
        def heuristic(chess_board, player):
            """
            Heuristic function to evaluate the board state.
            Higher scores indicate better states for the player.
            """
            # Number of discs owned
            player_discs = np.sum(chess_board == player)
            opponent_discs = np.sum(chess_board == (3 - player))

            # Corner positions are advantageous
            corners = [(0, 0), (0, chess_board.shape[1] - 1),
                       (chess_board.shape[0] - 1, 0), (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
            corner_score = sum(5 if chess_board[x, y] == player else -5 if chess_board[x, y] == (3 - player) else 0 for x, y in corners)

            # Total heuristic score
            return player_discs - opponent_discs + corner_score

        def simulated_annealing(chess_board, player, initial_temp, alpha, min_temp):
            """
            Simulated annealing for move selection.
            """
            current_board = deepcopy(chess_board)
            current_temp = initial_temp
            best_move = None

            while current_temp > min_temp:
                valid_moves = get_valid_moves(current_board, player)
                if not valid_moves:
                    break

                # Randomly choose a neighbor state (valid move)
                move = valid_moves[np.random.randint(len(valid_moves))]
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

            return best_move

        # Run simulated annealing to select the move
        best_move = simulated_annealing(chess_board, player, initial_temp, alpha, min_temp)

        # Ensure move selection fits within the time limit
        time_taken = time.time() - start_time
        if time_taken >= 2.0:
            print("Time limit approaching, returning best move found so far.")

        print("My AI's turn took ", time_taken, "seconds.")

        # If no move was found (unlikely), fall back to a random move
        return best_move if best_move else random_move(chess_board, player)
