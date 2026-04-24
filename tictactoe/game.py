# Standard Library Imports
from dataclasses import dataclass, field
from typing import Optional

# Third-Party Library Imports
from fastapi import WebSocket

# Local Application Imports

@dataclass
class Game:
    game_id: str
    players: list[WebSocket] = field(default_factory=list)   # [X_socket, O_socket]
    board: list[str] = field(default_factory=lambda: [""] * 9)
    current_turn: int = 0  # index into players list
    winner: Optional[str] = None
    game_over: bool = False

    WINNING_COMBOS = [
        [0,1,2],[3,4,5],[6,7,8],  # rows
        [0,3,6],[1,4,7],[2,5,8],  # cols
        [0,4,8],[2,4,6],          # diags
    ]

    def make_move(self, player_index: int, cell: int) -> bool:
        """Returns True if the move is valid and applied."""
        if self.game_over:
            return False
        if self.current_turn != player_index:
            return False
        if cell < 0 or cell > 8 or self.board[cell] != "":
            return False

        symbol = "X" if player_index == 0 else "O"
        self.board[cell] = symbol
        self._check_winner(symbol)
        if not self.game_over:
            self.current_turn = 1 - self.current_turn  # swap turns
        return True

    def _check_winner(self, symbol: str):
        for combo in self.WINNING_COMBOS:
            if all(self.board[i] == symbol for i in combo):
                self.winner = symbol
                self.game_over = True
                return
        if all(cell != "" for cell in self.board):
            self.game_over = True   # draw, winner stays None