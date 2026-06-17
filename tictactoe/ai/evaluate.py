"""
State-space evaluation of a trained QAgent against Minimax.

For every legal non-terminal board position where the outcome is not
forced (i.e. at least 2 of {win, draw, lose} are reachable), we:
  1. Ask Minimax for all moves that share the highest score.
  2. Ask the agent (greedy) for its best move.
  3. Score the agent's move using Minimax's scoring function.

Then we report aggregate statistics.

Run from the /tictactoe directory:
    python -m ai.evaluate --model ai/models/<your_model>.pkl
"""

# Standard Library Imports
import argparse
import statistics
from collections import Counter
from itertools import product
from pathlib import Path

# Local Application Imports
from ai.minimax import minimax, get_winner
from ai.q_agent import QAgent


# ---------------------------------------------------------------------------
# Board generation
# ---------------------------------------------------------------------------

WINNING_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6],
]


def whose_turn(board: tuple) -> str:
    """X always goes first; counts determine whose move it is."""
    xs = board.count("X")
    os = board.count("O")
    return "X" if xs == os else "O"


def is_legal(board: tuple) -> bool:
    """
    A board is legal if:
    - X count is either equal to O count (X to move) or one more (O to move)
    - At most one winner exists, and if there is one the game would have
      already ended (i.e. after that winner appeared no more moves were made).
    """
    xs = board.count("X")
    os = board.count("O")
    if xs < os or xs > os + 1:
        return False

    winner = get_winner(list(board))

    if winner and winner != "draw":
        # A winner means the game is over — this is a terminal state,
        # not a state we can act from. Exclude it.
        return False

    return True


def all_legal_boards() -> list[tuple]:
    """
    Enumerate every legal, non-terminal board position.
    There are 3^9 = 19,683 raw combinations; legal non-terminal ones are ~5,477.
    """
    legal = []
    for cells in product(["", "X", "O"], repeat=9):
        if is_legal(cells):
            legal.append(cells)
    return legal


# ---------------------------------------------------------------------------
# Outcome reachability — is the position "interesting"?
# ---------------------------------------------------------------------------

def reachable_outcomes(board: tuple, mover: str) -> set[str]:
    """
    Returns the set of outcomes reachable from this position under optimal
    and sub-optimal play.  Values: "win", "lose", "draw"
    (from the mover's perspective).
    """
    opp = "O" if mover == "X" else "X"
    seen: set[str] = set()

    def dfs(b: list, turn: str):
        if len(seen) == 3:   # short-circuit once all outcomes found
            return
        w = get_winner(b)
        if w == mover:
            seen.add("win")
            return
        if w == opp:
            seen.add("lose")
            return
        if w == "draw":
            seen.add("draw")
            return
        for i, cell in enumerate(b):
            if cell == "":
                b[i] = turn
                dfs(b, opp if turn == mover else mover)
                b[i] = ""
                if len(seen) == 3:
                    return

    dfs(list(board), mover)
    return seen


# ---------------------------------------------------------------------------
# Minimax scoring for every legal move from a position
# ---------------------------------------------------------------------------

def minimax_scores(board: tuple, mover: str) -> dict[int, int]:
    """Returns {cell: minimax_score} for every empty cell."""
    scores = {}
    b = list(board)
    for i, cell in enumerate(b):
        if cell == "":
            b[i] = mover
            scores[i] = minimax(b, False, mover, depth=1)
            b[i] = ""
    return scores


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: QAgent, verbose: bool = False) -> None:
    agent.epsilon = 0.0     # fully greedy

    boards = all_legal_boards()
    print(f"Total legal non-terminal positions : {len(boards)}")

    # Filter: only positions where outcome is not forced
    interesting = []
    for board in boards:
        mover = whose_turn(board)
        outcomes = reachable_outcomes(board, mover)
        if len(outcomes) >= 2:
            interesting.append((board, mover))

    print(f"Non-forced positions (>= 2 outcomes reachable) : {len(interesting)}")
    print()

    # ---------------------------------------------------------------------------
    # Per-state scoring
    # ---------------------------------------------------------------------------

    # For each state we record:
    #   minimax_best  : the highest minimax score achievable
    #   agent_score   : minimax score of the agent's chosen move
    #   is_optimal    : whether agent_score == minimax_best

    records = []

    for board, mover in interesting:
        mm_scores = minimax_scores(board, mover)
        mm_best   = max(mm_scores.values())
        mm_best_moves = [c for c, s in mm_scores.items() if s == mm_best]

        # Agent picks greedily
        canon = agent.canonicalize(list(board), mover)
        legal = [i for i, c in enumerate(board) if c == ""]
        agent_move = max(legal, key=lambda a: agent._q(canon, a))

        agent_score = mm_scores[agent_move]

        records.append({
            "board":          board,
            "mover":          mover,
            "mm_best":        mm_best,
            "mm_best_moves":  mm_best_moves,
            "agent_move":     agent_move,
            "agent_score":    agent_score,
            "is_optimal":     agent_score == mm_best,
        })

        if verbose:
            tag = "✓" if agent_score == mm_best else "✗"
            print(
                f"{tag} mover={mover} "
                f"mm_best={mm_best:+d} (cells {mm_best_moves})  "
                f"agent_move={agent_move} agent_score={agent_score:+d}"
            )

    # ---------------------------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------------------------

    n = len(records)
    agent_scores  = [r["agent_score"]  for r in records]
    mm_best_scores = [r["mm_best"]     for r in records]
    optimal_count  = sum(r["is_optimal"] for r in records)

    def stats_block(label: str, values: list[int]) -> None:
        mode_val = Counter(values).most_common(1)[0][0]
        print(f"  {label}")
        print(f"    highest  : {max(values):+d}")
        print(f"    lowest   : {min(values):+d}")
        print(f"    mean     : {statistics.mean(values):+.4f}")
        print(f"    median   : {statistics.median(values):+.1f}")
        print(f"    mode     : {mode_val:+d}")
        cumulative = sum(values)
        print(f"    cumulative sum : {cumulative:+d}")
        print()

    print("=" * 60)
    print(f"RESULTS  ({n} non-forced states evaluated)")
    print("=" * 60)
    print()

    # Theoretical ceiling — if every move were optimal
    print("Minimax best-move scores (theoretical ceiling)")
    print("-" * 45)
    stats_block("per-state minimax best score", mm_best_scores)

    print("Agent scores (minimax value of agent's chosen move)")
    print("-" * 45)
    stats_block("per-state agent score", agent_scores)

    # Optimality rate
    print(f"Optimal moves (agent score == minimax best)")
    print(f"  {optimal_count} / {n}  ({optimal_count/n:.1%})")
    print()

    # Cumulative score gap
    ceiling   = sum(mm_best_scores)
    achieved  = sum(agent_scores)
    gap       = ceiling - achieved
    print(f"Cumulative score gap (ceiling − agent) : {gap:+d}")
    print(f"  ceiling  : {ceiling:+d}")
    print(f"  achieved : {achieved:+d}")
    print()

    # Breakdown by move-score bucket
    print("Agent score distribution")
    print("-" * 45)
    score_dist = Counter(agent_scores)
    for score in sorted(score_dist, reverse=True):
        pct = score_dist[score] / n
        bar = "█" * int(pct * 40)
        label = (
            "win  (immediate)"  if score >= 9  else
            "win  (delayed)"    if score > 0   else
            "draw"              if score == 0  else
            "lose (delayed)"    if score > -9  else
            "lose (immediate)"
        )
        print(f"  {score:+3d}  {label:<22}  {score_dist[score]:>5}  ({pct:.1%})  {bar}")
    print()

    # Suboptimal move breakdown
    suboptimal = [r for r in records if not r["is_optimal"]]
    if suboptimal:
        print(f"Suboptimal move breakdown  ({len(suboptimal)} states)")
        print("-" * 45)
        gap_dist = Counter(r["mm_best"] - r["agent_score"] for r in suboptimal)
        for gap_val in sorted(gap_dist):
            print(f"  missed by {gap_val:+d} : {gap_dist[gap_val]} states")
        print()

        print("Sample suboptimal states (up to 5)")
        print("-" * 45)
        for r in suboptimal[:5]:
            board = r["board"]
            rows = []
            for row in range(3):
                cells = [board[row*3+col] or "·" for col in range(3)]
                rows.append(" ".join(cells))
            print(f"  mover={r['mover']}  mm_best={r['mm_best']:+d} (cells {r['mm_best_moves']})  "
                  f"agent chose {r['agent_move']} (score {r['agent_score']:+d})")
            for row in rows:
                print(f"    {row}")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained QAgent against Minimax.")
    parser.add_argument("--model", required=True, help="Path to .pkl Q-table file")
    parser.add_argument("--verbose", action="store_true", help="Print every state result")
    args = parser.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"Model file not found: {path}")
        return

    agent = QAgent()
    agent.load(path)

    evaluate(agent, verbose=args.verbose)


if __name__ == "__main__":
    main()