from agents.agent import Agent
from store import register_agent
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves
import numpy as np
import time


@register_agent("james_agent")
class JamesAgent(Agent):
    def __init__(self):
        super(JamesAgent, self).__init__()
        self.name = "james_agent"
        self.transposition_table = {}  # Hash map for storing game states
        self.max_table_size = 90320
        """
        heuristic approach to dynamically determine the size of 
        the transposition table based on the board size

        max table size should be board_size^4 * 10 (heuristic)
        but since i cant figure it out, 
            6: 12960  max entries
            8: 40960
            10: 100000
            12: 207360  max entries: 

            avg ~= 90320
        """

    def step(self, chess_board, player, opponent):
        """
        Decide the next move using iterative deepening and alpha-beta pruning.
        """
        start_time = time.time()
        time_limit = 1.99
        best_move = None
        depth = 1
        

        while True:
            try:
                best_move = self.minimax(chess_board, player, opponent, start_time, time_limit, depth)
                depth += 1
            except TimeoutError:
                break

        if best_move is None:
            best_move = random_move(chess_board, player)

        return best_move

    def minimax(self, chess_board, player, opponent, start_time, time_limit, max_depth):
        """
        Perform iterative deepening search with alpha-beta pruning and state memorization.
        """
        board_hash = hash(chess_board.tobytes())
        if board_hash in self.transposition_table:
            cached_state = self.transposition_table[board_hash]
            if cached_state["depth"] >= max_depth:
                return cached_state["best_move"]

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None

        best_move = None
        alpha = -float("inf")
        beta = float("inf")

        for move in valid_moves:
            if time.time() - start_time >= time_limit:
                raise TimeoutError

            sim_board = chess_board.copy()
            execute_move(sim_board, move, player)

            move_score = self.alpha_beta(
                sim_board, max_depth - 1, alpha, beta, False, player, opponent, start_time, time_limit
            )

            if move_score > alpha:
                alpha = move_score
                best_move = move

        self.store_in_table(board_hash, best_move, alpha, beta, max_depth)
        return best_move

    def alpha_beta(self, board, depth, alpha, beta, is_maximizing, player, opponent, start_time, time_limit):
        """
        Perform alpha-beta pruning with state memorization at leaf nodes.
        """
        board_hash = hash(board.tobytes())
        if board_hash in self.transposition_table:
            cached_state = self.transposition_table[board_hash]
            if cached_state["depth"] >= depth:
                return cached_state["alpha"] if is_maximizing else cached_state["beta"]

        if time.time() - start_time >= time_limit:
            raise TimeoutError

        is_endgame, p1_score, p2_score = check_endgame(board, player, opponent)
        if is_endgame:
            return p1_score - p2_score if player == 1 else p2_score - p1_score
        if depth == 0:
            return self.heuristic_score(board, player, opponent)

        valid_moves = get_valid_moves(board, player if is_maximizing else opponent)
        if not valid_moves:
            return self.alpha_beta(board, depth - 1, alpha, beta, not is_maximizing, player, opponent, start_time, time_limit)

        if is_maximizing:
            max_eval = -float("inf")
            for move in valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError

                sim_board = board.copy()
                execute_move(sim_board, move, player)

                eval_score = self.alpha_beta(sim_board, depth - 1, alpha, beta, False, player, opponent, start_time, time_limit)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Prune
            self.store_in_table(board_hash, None, max_eval, beta, depth)
            return max_eval
        else:
            min_eval = float("inf")
            for move in valid_moves:
                if time.time() - start_time >= time_limit:
                    raise TimeoutError

                sim_board = board.copy()
                execute_move(sim_board, move, opponent)

                eval_score = self.alpha_beta(sim_board, depth - 1, alpha, beta, True, player, opponent, start_time, time_limit)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Prune
            self.store_in_table(board_hash, None, alpha, min_eval, depth)
            return min_eval

    def heuristic_score(self, board, player, opponent):
        """
        Heuristic evaluation of the board state.
        """
        player_score = np.sum(board == player)
        opponent_score = np.sum(board == opponent)
        mobility = len(get_valid_moves(board, player)) - len(get_valid_moves(board, opponent))
        corners = [(0, 0), (0, len(board) - 1), (len(board) - 1, 0), (len(board) - 1, len(board) - 1)]
        corner_score = sum(int(board[x, y] == player) - int(board[x, y] == opponent) for x, y in corners)

        return 2 * (player_score - opponent_score) + 3 * mobility + 5 * corner_score

    def store_in_table(self, board_hash, best_move, alpha, beta, depth):
        """
        Store state in the transposition table with a size limit.
        """
        if len(self.transposition_table) > self.max_table_size:
            self.transposition_table.clear()  # Clear table if size limit exceeded

        self.transposition_table[board_hash] = {"best_move": best_move, "alpha": alpha, "beta": beta, "depth": depth}

