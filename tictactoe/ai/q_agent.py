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

    def act(self, board: list[str], my_symbol: str) -> tuple[tuple, int]:
        state = self.canonicalize(board, my_symbol)

        legal = [i for i, cell in enumerate(board) if cell == ""]

        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            action = self._best_action(state, legal)

        return state, action

    def learn(
        self,
        state: tuple,
        action: int,
        next_board: list[str],
        my_symbol: str,
        reward: float,
        done: bool,
    ):
        next_state = self.canonicalize(next_board, my_symbol)

        legal_next = [
            i
            for i, cell in enumerate(next_board)
            if cell == ""
        ]

        if done or not legal_next:
            future = 0.0
        else:
            future = max(
                self._q(next_state, a)
                for a in legal_next
            )

        current = self._q(state, action)

        self.q_table[(state, action)] = (
            current
            + self.alpha
            * (
                reward
                + self.gamma * future
                - current
            )
        )

    def decay_epsilon(self):
        """Step epsilon down after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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