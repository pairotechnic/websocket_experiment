import pickle
import random
from pathlib import Path

class QAgent:
    def __init__(
        self,
        symbol: str,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995
    ):
        self.symbol = symbol
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: dict[tuple, float] = {}

        # Memory of the last (state, action) so we can update Q after seeing the result
        self.last_state: tuple | None = None
        self.last_action: int | None = None

    # Q-table helpers -----------------

    def _q(self, state: tuple, action: int) -> float:
        return self.q_table.get((state, action), 0.0)
    
    def _best_action(self, state: tuple, legal: list[int]) -> int:
        return max(legal, key=lambda a: self._q(state, a))
    
    # Core interface -------------------

    def act(self, board: list[str]) -> int:
        """
        Choose an action using ε-greedy policy.
        Saves (state, action) so learn() can reference them.
        """
        state = tuple(board)
        legal = [i for i, cell in enumerate(board) if cell == ""]

        if random.random() < self.epsilon:
            action = random.choice(legal)
        else :
            action = self._best_action(state, legal)

        self.last_state = state
        self.last_action = action
        return action
    
    def learn(self, board: list[str], reward: float, done: bool):
        """
        Update Q(last_state, last_action) using the Bellman equation.
        Call this after every move, passing the resulting board and award.
        """
        # ANSWER : When is self.last_state ever going to be None?
        if self.last_state is None: 
            return
        
        next_state = tuple(board)
        legal_next = [i for i, cell in enumerate(board) if cell == ""]

        if done or not legal_next:
            future = 0.0
        else :
            future = max(self._q(next_state, a) for a in legal_next)

        current = self._q(self.last_state, self.last_action)
        self.q_table[(self.last_state, self.last_action)] = (
            current + self.alpha * (reward + self.gamma * future - current)
        )

    def decay_epsilon(self):
        """Step epsilon down after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence --------------------------

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Saved Q-table ({len(self.q_table)} entries -> {path})")

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Loaded Q-table ({len(self.q_table)} entries) <- {path}")