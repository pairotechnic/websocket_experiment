# Standard Library Imports
from pathlib import Path
import pickle
import random

# Third-Party Library Imports

# Local Application Imports


class QAgent:
    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.99997,
    ):
        # No symbol stored here — the agent always reasons from the perspective
        # of "my symbol = X, opponent symbol = O" via board canonicalization.
        # The caller is responsible for passing the right symbol to act/learn.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict[tuple, float] = {}

        # Memory of the last (state, action) so we can update Q after seeing
        # the result of our move (which only becomes visible on the next turn).
        self.last_state: tuple | None = None
        self.last_action: int | None = None

    # ------------------------------------------------------------------
    # Board canonicalization
    # ------------------------------------------------------------------

    @staticmethod
    def canonicalize(board: list[str], my_symbol: str) -> tuple:
        """
        Return the board as a tuple from the agent's own perspective:
          my_symbol  -> 'X'
          opponent   -> 'O'
          empty      -> ''
        This lets a single Q-table serve both X and O players.
        """
        opp = "O" if my_symbol == "X" else "X"
        return tuple(
            "X" if cell == my_symbol else ("O" if cell == opp else "")
            for cell in board
        )

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------

    def _q(self, state: tuple, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def _best_action(self, state: tuple, legal: list[int]) -> int:
        return max(legal, key=lambda a: self._q(state, a))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def act(self, board: list[str], my_symbol: str) -> int:
        """
        Choose an action using an ε-greedy policy.
        Saves (canonical_state, action) so learn() can reference them.

        Args:
            board:     The raw game board (list of 'X', 'O', '').
            my_symbol: The symbol this agent is playing as this turn.

        Returns:
            Cell index (0-8) to play.
        """
        state = self.canonicalize(board, my_symbol)
        legal = [i for i, cell in enumerate(board) if cell == ""]

        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            action = self._best_action(state, legal)

        self.last_state = state
        self.last_action = action
        return action

    def learn(self, board: list[str], my_symbol: str, reward: float, done: bool):
        """
        Update Q(last_state, last_action) using the Bellman equation.

        Call this after every move, passing the board that *resulted* from
        the last action, along with the reward signal and whether the game ended.

        Args:
            board:     The raw board state after the move was applied.
            my_symbol: The symbol this agent was playing when it made the move.
            reward:    Reward signal (+1 win, -1 loss, +0.5 draw, 0 otherwise).
            done:      True if the episode is over.
        """
        if self.last_state is None:
            return

        next_state = self.canonicalize(board, my_symbol)
        legal_next = [i for i, cell in enumerate(board) if cell == ""]

        if done or not legal_next:
            future = 0.0
        else:
            future = max(self._q(next_state, a) for a in legal_next)

        current = self._q(self.last_state, self.last_action)
        self.q_table[(self.last_state, self.last_action)] = (
            current + self.alpha * (reward + self.gamma * future - current)
        )

    def decay_epsilon(self):
        """Step epsilon down after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        """Clear per-episode memory. Call at the start of each new game."""
        self.last_state = None
        self.last_action = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Saved Q-table ({len(self.q_table)} entries -> {path})")

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Loaded Q-table ({len(self.q_table)} entries) <- {path}")