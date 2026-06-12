# Standard Library Imports
import asyncio

# Third-Party Library Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Local Application Imports
from manager import GameManager, AIGameManager
from ai.minimax import MinimaxAgent


app = FastAPI()
manager = GameManager()

# HELPERS
async def _read_loop(ws: WebSocket, game_id: str, player_index: int):
    async for data in ws.iter_json():
        if data.get("type") == "move":
            await manager.handle_move(game_id, player_index, data["cell"])

# ENDPOINTS - pages
@app.get("/tictactoe")
async def get_lobby():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.get("/tictactoe/random")
async def get_random_game():
    with open("game.html") as f:
        return HTMLResponse(f.read())
    
@app.get("/tictactoe/minimax")
async def get_minimax_game():
    with open("game.html") as f:
        return HTMLResponse(f.read())

# ENDPOINTS - websockets
@app.websocket("/tictactoe/ws/minimax/{player_id}")
async def minimax_ws(ws: WebSocket, player_id: str, symbol: str = "X"):
    """Human vs Minimax - Symbol passed as query param ?symbol=X or ?symbol=O."""
    agent = MinimaxAgent()
    mgr = AIGameManager(agent)
    await mgr.run(player_id, ws, human_symbol=symbol)

@app.websocket("/tictactoe/ws/{player_id}")
async def random_ws(ws: WebSocket, player_id: str):
    """Human vs Human"""
    result = await manager.connect(player_id, ws)
    if not result:
        return
    game_id, player_index = result

    reader_task = asyncio.create_task(
        _read_loop(ws, game_id, player_index)
    )
    manager.register_task(game_id, player_index, reader_task)  # register after creation

    try:
        await reader_task
    except asyncio.CancelledError:
        pass
    finally:
        await manager.disconnect(game_id, player_index)