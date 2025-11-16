from .utils import chess_manager, GameContext
import chess
import torch
import time
import math
from pathlib import Path

from .nn_model import ValueNet, board_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = Path(__file__).with_name("valuenet.pt")

# ------------------------
# Load neural network
# ------------------------

try:
    _model = ValueNet().to(DEVICE)
    _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    _model.eval()
    print(f"[INFO] Loaded neural network weights from {WEIGHTS_PATH}")
except Exception as e:
    print(f"[WARNING] Could not load weights ({e}) — using RANDOM model (will play badly)")
    _model = ValueNet().to(DEVICE)
    _model.eval()


@torch.no_grad()
def evaluate(board: chess.Board) -> float:
    """
    Pure neural evaluation.
    Output squashed to [-1, 1].
    Positive = good for White, negative = good for Black.
    """
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    v = _model(x)          # (1,1)
    return torch.tanh(v).item()


INF = 1e9

# ------------------------
# Search
# ------------------------

def eval_terminal(board: chess.Board) -> float:
    """
    Exact outcome for terminal positions, otherwise NN.
    Always from White's perspective.
    """
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0
    return evaluate(board)


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """
    Standard alpha-beta search using the NN as leaf evaluator.
    Returns value from White's perspective.
    """
    if depth == 0 or board.is_game_over():
        return eval_terminal(board)

    if board.turn == chess.WHITE:
        value = -INF
        for move in board.legal_moves:
            board.push(move)
            value = max(value, alpha_beta(board, depth - 1, alpha, beta))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for move in board.legal_moves:
            board.push(move)
            value = min(value, alpha_beta(board, depth - 1, alpha, beta))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def search_root(board: chess.Board, depth: int):
    """
    Search all legal moves to given depth and pick the best one.
    Returns (best_move, scores_in_move_order, moves_in_same_order).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, [], []

    scores = []
    white_to_move = board.turn == chess.WHITE
    best_score = -INF if white_to_move else INF
    best_move = legal_moves[0]

    for move in legal_moves:
        board.push(move)
        score = alpha_beta(board, depth - 1, -INF, INF)
        board.pop()
        scores.append(score)

        if white_to_move and score > best_score:
            best_score = score
            best_move = move
        if not white_to_move and score < best_score:
            best_score = score
            best_move = move

    return best_move, scores, legal_moves


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    live_board = ctx.board

    if live_board.is_game_over():
        ctx.logProbabilities({})
        raise ValueError("Game over")

    # Work on a COPY so we don't mutate ctx.board during search
    board_for_search = live_board.copy()

    # Reduce depth a bit to avoid timeouts
    SEARCH_DEPTH = 2   # you can try 3 later if it's fast enough

    # Run search (only once)
    best_move, scores, root_moves = search_root(board_for_search, depth=SEARCH_DEPTH)

    if best_move is None or not root_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Map scores to moves from the root search
    move_to_score = {m: s for m, s in zip(root_moves, scores)}

    # Legal moves for the REAL board
    legal_moves = list(live_board.legal_moves)

    # Ensure chosen move is legal in the live position
    if best_move not in legal_moves:
        # fallback: pick best legal move according to move_to_score
        scored_legal = [(m, move_to_score.get(m, 0.0)) for m in legal_moves]
        white_to_move = live_board.turn == chess.WHITE
        if white_to_move:
            best_move, _ = max(scored_legal, key=lambda ms: ms[1])
        else:
            best_move, _ = min(scored_legal, key=lambda ms: ms[1])

    # Build probabilities using the *existing* scores, no extra alpha-beta calls
    white_to_move = live_board.turn == chess.WHITE
    raw_scores = []
    for m in legal_moves:
        s = move_to_score.get(m, 0.0)
        raw_scores.append(s if white_to_move else -s)

    if raw_scores:
        max_s = max(raw_scores)
        exps = [math.exp(s - max_s) for s in raw_scores]
        total = sum(exps)
        if total <= 0:
            probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            probs = [e / total for e in exps]
    else:
        probs = [1.0 / len(legal_moves)] * len(legal_moves)

    ctx.logProbabilities({m: p for m, p in zip(legal_moves, probs)})

    print(f"Depth {SEARCH_DEPTH}, chosen {best_move}")
    # time.sleep(0.02)  # remove this delay – it just slows you down
    return best_move
