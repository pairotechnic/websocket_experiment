# Standard Library Imports
import asyncio
import uuid

# Third-Party Library Imports
from fastapi import WebSocket

# Local Application Imports
from game import Game

class GameManager:
    def __init__(self):
        self.waiting: asyncio.Queue[tuple[str, WebSocket, asyncio.Future[tuple[str, int]]]] = asyncio.Queue()
        self.games: dict[str, Game] = {}
        self.reader_tasks: dict[str, list[asyncio.Task]] = {}  # game_id -> [task_X, task_O]

    def register_task(self, game_id: str, player_index: int, task: asyncio.Task):
        if game_id not in self.reader_tasks:
            self.reader_tasks[game_id] = [None, None]
        self.reader_tasks[game_id][player_index] = task

    async def connect(self, player_id: str, ws: WebSocket) -> tuple[str, int] | None:
        await ws.accept()

        loop = asyncio.get_running_loop()
        matched: asyncio.Future = loop.create_future()

        # Put this player in the queue
        await self.waiting.put((player_id, ws, matched))
        await ws.send_json({"type": "waiting", "message": "Looking for opponent..."})

        # If there are now 2 waiting players, start a game
        if self.waiting.qsize() >= 2:
            pid1, ws1, fut1 = await self.waiting.get()
            pid2, ws2, fut2 = await self.waiting.get()
            await self._start_game(pid1, ws1, fut1, pid2, ws2, fut2)
        
        # Both players await their own fututre - resolves when _start_game sets the result
        return await matched

    # Returns which index *this* socket is
    async def _start_game(
        self, 
        pid1: str, ws1: WebSocket, fut1: asyncio.Future,
        pid2: str, ws2: WebSocket, fut2: asyncio.Future
    ):
        
        game_id = str(uuid.uuid4())[:8]
        game = Game(game_id=game_id, players=[ws1, ws2], player_ids=[pid1, pid2])
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

        fut1.set_result((game_id, 0)) # X is index 0
        fut2.set_result((game_id, 1)) # O is index 1

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

        other = 1 - player_index
        # Notify the other player
        try:
            await game.players[other].send_json({
                "type": "game_over",
                "winner": None,
                "message": "Opponent disconnected.",
            })
        except Exception:
            pass

        # Cancel the other player's reader task — unblocks them immediately
        tasks = self.reader_tasks.pop(game_id, [None, None])
        other_task = tasks[other]
        if other_task and not other_task.done():
            other_task.cancel()

        del self.games[game_id]