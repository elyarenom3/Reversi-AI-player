# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("three_step_agent")
class three_step_Agent(Agent):

  def __init__(self):
    super(three_step_Agent, self).__init__()
    self.name = "three_step_agent"
    self.autoplay = True

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    """
    IDEAS:

      - MTCS: Tree policy: Sim Policy:
      - Simulated Annealing: 

      - Greedy?
      - GPT_greedy_corner works with score and combines multiple methods. 
      
      # IDEA: Implement Minimax, then MCTS, then Endgame special solver (minimax again?)

    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    # give ourselves a buffer of 0.1 seconds to return.
    time_limit_to_think = 1.99



    move = self.min_max_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think)







    print("My AI's turn took ", time_taken, "seconds.")


    if (move == None) :
      print('smt went wrong and min_max didnt return :( giving random move...')
      move = random_move(chess_board,player)

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return move

  def min_max_give_me_ur_best_move(self, chess_board, player, start_time, time_limit):

    ops = 3 - player 
    
    valid_moves = get_valid_moves(chess_board, player)

    best_move = None
    best_score = -float('inf')

    alpha = -float('inf')
    beta = float('inf')

     # start at depth 1
    depth = 1
    # We want to do IDS with time search, with a buffer of maybe 0.1 seconds to make sure we return so
    while True : 
      try: 
          for move in valid_moves:
            # If we're out of time then raise an error
            if time.time() - start_time > time_limit: raise TimeoutError

            # to try moves we create a copy of the chess_board
            sim_board = deepcopy(chess_board)
            # try the move
            execute_move(sim_board, move, player)

            eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit)

            if eval_score > best_score:
              best_score = eval_score
              best_move = move

            # update alpha
            alpha = max(alpha, eval_score)

            # Prune
            if beta <= alpha:
                break
            
          # (outside the for loop) augment depth for IDS
          depth += 1

      except TimeoutError:
        break # if we dont have time left, then we return the best we have for now
    return best_move
  def min_max_score(self, chess_board, depth, alpha, beta, max_or_nah, player, ops, start_time, time_limit):
      # Check if time is short and return if we dont have time anymore
      if time.time() - start_time > time_limit:
          raise TimeoutError

      # If the game has ended, calculate the score and return it
      (is_end, p1_score, p2_score) = check_endgame(chess_board, player, ops)
      if is_end:
          if player == 1:
              return p1_score - p2_score
          else:
              return p2_score - p1_score

      # If we're at depth 0, end recursion and return heuristic score
      if depth == 0:
          return self.heuristic_score(chess_board, player, ops)

      # Get valid moves for the current player
      valid_moves = get_valid_moves(chess_board, player if max_or_nah else ops)

      # If no valid moves, switch turns but decrement depth
      if len(valid_moves) == 0:
          not_max_or_nah = not max_or_nah
          return self.min_max_score(chess_board, depth - 1, alpha, beta, not_max_or_nah, player, ops, start_time, time_limit)

      # Determine the player for this step
      current_player = player if max_or_nah else ops

      # Maximizing step
      if max_or_nah:
          max_eval = -float('inf')
          for move in valid_moves:
              # Check if time is short and return if we dont have time anymore
              if time.time() - start_time > time_limit:
                  raise TimeoutError

              # Try the move on a copy of the board
              sim_board = deepcopy(chess_board)
              execute_move(sim_board, move, current_player)

              # Recur to minimize the next step
              eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit)

              # Maximize the score
              max_eval = max(max_eval, eval_score)

              # Update alpha
              alpha = max(alpha, eval_score)

              # Prune
              if beta <= alpha:
                  break
          return max_eval

      # Minimizing step
      else:
          min_eval = float('inf')
          for move in valid_moves:
              # Check if time is short and return if we dont have time anymore
              if time.time() - start_time > time_limit:
                  raise TimeoutError

              # Try the move on a copy of the board
              sim_board = deepcopy(chess_board)
              execute_move(sim_board, move, current_player)

              # Recur to maximize the next step
              eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, True, player, ops, start_time, time_limit)

              # Minimize the score
              min_eval = min(min_eval, eval_score)

              # Update beta
              beta = min(beta, eval_score)

              # Prune
              if beta <= alpha:
                  break
          return min_eval



  def heuristic_score(self, chess_board, player, ops):
      # count the number of brown and blue, simple greedy way to evaluate, could maybe use helper, but oh well, works.. find better heuristic?
      player_score = np.sum(chess_board == player)
      opponent_score = np.sum(chess_board == ops)
      return player_score - opponent_score