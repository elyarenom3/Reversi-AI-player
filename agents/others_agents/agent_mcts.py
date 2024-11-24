##### TEMP CODE UNTIL WE FINISH STUDENT AGENT #####

# Attempt 1)

# Student agent: Add your own agent here DONT FORGET TO REMOVE THE GAME. bcs its only for our folder structure

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("agent_mcts")
class agent_mcts(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(agent_mcts, self).__init__()
    self.name = "agent_mcts"

    def step(self, chess_board, player, opponent):
        """
        Decide the next move using MCTS.
        """
        def get_base_move_scores(board, player, moves):
            scores = np.zeros(len(moves))
            for i, move in enumerate(moves):
                simulated_board = deepcopy(board)
                execute_move(simulated_board, move, player)
                _, player_score, opp_score = check_endgame(simulated_board, player, opponent)
                scores[i] = player_score - opp_score
            return scores

        def swap_players(current, other):
            return other, current

        def simulate_to_end(board, p, q):
            """
            Simulate the game to completion using random moves for both players.
            """
            is_endgame, p_score, opp_score = check_endgame(board, player, opponent)
            while not is_endgame:
                move = random_move(board, p)
                if move is None:
                    p, q = swap_players(p, q)
                    move = random_move(board, p)
                execute_move(board, move, p)
                p, q = swap_players(p, q)
                is_endgame, p_score, opp_score = check_endgame(board, player, opponent)
            return p_score - opp_score

        def compute_move_scores(exploit, explore, n_parent, exploration_factor):
            return exploit + exploration_factor * np.sqrt(n_parent / (1 + explore))

        # Timing setup
        start_time = time.time()

        # Hyperparameters
        exploration_factor = 1.4  # Exploration parameter for UCT
        max_simulation_time = 1.99  # Maximum time for simulations

        # Get valid moves
        possible_moves = get_valid_moves(chess_board, player)
        if len(possible_moves) == 1:
            return possible_moves[0]

        # Initialize MCTS tree
        tree_states = [(chess_board, player)]  # (board_state, current_player)
        node_moves = [possible_moves]
        exploit = [get_base_move_scores(chess_board, player, possible_moves)]
        explore = [np.ones(len(possible_moves))]  # Exploration count
        node_visits = [1]

        while time.time() - start_time < max_simulation_time:
            simulated_board = deepcopy(chess_board)
            current_player, other_player = player, opponent
            depth = 0
            path_indices = []  # Path of nodes explored
            move_indices = []  # Path of moves taken

            # Selection phase: Traverse the tree
            while True:
                # Find node index in the tree
                try:
                    node_index = next(
                        i for i, (state, pl) in enumerate(tree_states)
                        if np.array_equal(state, simulated_board) and pl == current_player
                    )
                except StopIteration:
                    node_index = -1

                if node_index == -1:
                    break  # Node not found, expand it

                # Get scores and pick the best move
                scores = compute_move_scores(
                    exploit[node_index], explore[node_index], node_visits[node_index], exploration_factor
                )
                move_index = np.argmax(scores)
                move = node_moves[node_index][move_index]
                if move is not None:
                    execute_move(simulated_board, move, current_player)

                # Record traversal path
                path_indices.append(node_index)
                move_indices.append(move_index)

                # Swap players
                current_player, other_player = swap_players(current_player, other_player)
                depth += 1

            # Expansion phase: Add a new node to the tree
            valid_moves = get_valid_moves(simulated_board, current_player)
            if valid_moves:
                tree_states.append((simulated_board, current_player))
                exploit.append(get_base_move_scores(simulated_board, current_player, valid_moves))
                explore.append(np.ones(len(valid_moves)))
                node_moves.append(valid_moves)
                node_visits.append(1)

            # Simulation phase: Play randomly to the end
            result = simulate_to_end(deepcopy(simulated_board), current_player, other_player)

            # Backpropagation phase: Update tree with simulation results
            for idx, move_idx in zip(reversed(path_indices), reversed(move_indices)):
                node_visits[idx] += 1
                explore[idx][move_idx] += 1
                exploit[idx][move_idx] += result

        # Choose the best move based on exploitation scores
        best_move_index = np.argmax(exploit[0])
        return possible_moves[best_move_index]
