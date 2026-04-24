# Standard Library Imports
import asyncio
import uuid

# Third-Party Library Imports
from fastapi import WebSocket

# Local Application Imports
from game import Game

class GameManager:
    def __init__(self):
        self.waiting: asyncio.Queue[tuple[str, WebSocket]] = asyncio.Queue()
        self.games: dict[str, Game] = {}

    async def connect(self, player_id: str, ws: WebSocket):
        await ws.accept()

        # Put this player in the queue
        await self.waiting.put((player_id, ws))
        await ws.send_json({"type": "waiting", "message": "Looking for opponent..."})

        # If there are now 2 waiting players, start a game
        if self.waiting.qsize() >= 2:
            pid1, ws1 = await self.waiting.get()
            pid2, ws2 = await self.waiting.get()
            await self._start_game(pid1, ws1, pid2, ws2)

    async def _start_game(self, pid1: str, ws1: WebSocket, pid2: str, ws2: WebSocket):
        game_id = str(uuid.uuid4())[:8]
        game = Game(game_id=game_id, players=[ws1, ws2])
        self.games[game_id] = game

        # Notify both players
        await ws1.send_json({
            "type": "game_start",
            "game_id": game_id,
            "symbol": "X",
            "your_turn": True,
        })
        await ws2.send_json({
            "type": "game_start",
            "game_id": game_id,
            "symbol": "O",
            "your_turn": False,
        })

    async def handle_move(self, game_id: str, player_index: int, cell: int):
        game = self.games.get(game_id)
        if not game:
            return

        valid = game.make_move(player_index, cell)
        if not valid:
            ws = game.players[player_index]
            await ws.send_json({"type": "error", "message": "Invalid move"})
            return

        # Broadcast updated board to both players
        update = {
            "type": "game_update",
            "board": game.board,
            "current_turn": game.current_turn,
        }
        for ws in game.players:
            await ws.send_json(update)

        # Check if game ended
        if game.game_over:
            result = {
                "type": "game_over",
                "winner": game.winner,   # None = draw
                "board": game.board,
            }
            for ws in game.players:
                await ws.send_json(result)
            del self.games[game_id]

    async def disconnect(self, game_id: str, player_index: int):
        game = self.games.get(game_id)
        if not game:
            return
        # Notify the other player
        other = 1 - player_index
        if other < len(game.players):
            try:
                await game.players[other].send_json({
                    "type": "game_over",
                    "winner": None,
                    "message": "Opponent disconnected",
                })
            except Exception:
                pass
        del self.games[game_id]