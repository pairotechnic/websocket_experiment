# Standard Library Imports
import asyncio
import uuid

# Third-Party Library Imports
from fastapi import WebSocket, WebSocketDisconnect

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

        await self.waiting.put((player_id, ws, matched))
        await ws.send_json({"type": "waiting", "message": "Looking for opponent..."})

        # Drain any stale (disconnected) waiters before trying to pair
        if self.waiting.qsize() >= 2:
            live = []
            while not self.waiting.empty():
                entry = self.waiting.get_nowait()
                pid, candidate_ws, fut = entry
                try:
                    await candidate_ws.send_json({"type": "waiting", "message": "Looking for opponent..."})
                    live.append(entry)
                except Exception:
                    # Socket is dead — discard it; its future is abandoned
                    if not fut.done():
                        fut.cancel()

            for entry in live:
                await self.waiting.put(entry)

        if self.waiting.qsize() >= 2:
            pid1, ws1, fut1 = await self.waiting.get()
            pid2, ws2, fut2 = await self.waiting.get()
            await self._start_game(pid1, ws1, fut1, pid2, ws2, fut2)

        try:
            return await matched
        except asyncio.CancelledError:
            return None

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

class AIGameManager:
    """
    Manages a single game between one human WebSocket and an AI agent.

    The agent must implement :
        act(board: list[str], symbol: str) -> int
    returning the cell index (0-8) the AI wants to play.
    """

    def __init__(self, agent):
        self.agent = agent

    async def run(self, player_id: str, ws: WebSocket, human_symbol: str):
        """
        Full cycle of one human-vs-AI game.
        Called directly from the route handler - no queue, no waiting.
        """
        await ws.accept()

        ai_symbol = "O" if human_symbol == "X" else "X"
        game_id = str(uuid.uuid4())[:8]
        game = Game(
            game_id=game_id,
            players=[ws, None],             # slot 1 is unused - AI has no socket
            player_ids=[player_id, "ai"]
        )

        # Assign player indices based on symbol choice
        # X is always index 0, O is always index 1 (matches Game.make_move logic)
        human_index = 0 if human_symbol == "X" else 1
        ai_index = 1 - human_index

        # Tell the human the game is starting
        await ws.send_json({
            "type": "game_start",
            "game_id": game_id,
            "symbol": human_symbol,
            "your_turn": human_symbol == "X",    # X always goes first
        })

        # If AI plays X, it moves first before the human has done anything
        if ai_index == 0:
            await self._do_ai_turn(ws, game, ai_symbol, ai_index)

        # Main read loop - receives moves from the human
        try :
            async for data in ws.iter_json():
                if game.game_over:
                    break
                if data.get("type") != "move":
                    continue

                cell = data["cell"]
                valid = game.make_move(human_index, cell)

                if not valid:
                    await ws.send_json({"type": "error", "message": "Invalid move"})
                    continue

                # Send update after human's move
                await ws.send_json({
                    "type": "game_update",
                    "board": game.board,
                    "current_turn": game.current_turn,
                })

                if game.game_over:
                    await ws.send_json({
                        "type": "game_over",
                        "winner": game.winner,
                        "board": game.board
                    })
                    break

                # AI's turn
                await self._do_ai_turn(ws, game, ai_symbol, ai_index)

        except WebSocketDisconnect:
            pass    # human closed the tab - nothing to clean up

    async def _do_ai_turn(
        self, 
        ws: WebSocket,
        game: Game,
        ai_symbol: str,
        ai_index: int
    ):
        """Compute AI move, apply it, broadcast result."""
        cell = self.agent.act(game.board, ai_symbol)
        game.make_move(ai_index, cell)

        await ws.send_json({
            "type": "game_update",
            "board": game.board,
            "current_turn": game.current_turn,
        })

        if game.game_over:
            await ws.send_json({
                "type": "game_over",
                "winner": game.winner,
                "board": game.board
            })

