# Standard Library Imports
import asyncio

# Third-Party Library Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Local Application Imports
from manager import GameManager


app = FastAPI()
manager = GameManager()

# HELPERS
async def _read_loop(ws: WebSocket, game_id: str, player_index: int):
    async for data in ws.iter_json():
        if data.get("type") == "move":
            await manager.handle_move(game_id, player_index, data["cell"])

# ENDPOINTS
@app.get("/tictactoe")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/tictactoe/ws/{player_id}")
async def websocket_endpoint(ws: WebSocket, player_id: str):
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

@app.get("/tictactoe/random")
async def get_game():
    with open("game.html") as f:
        return HTMLResponse(f.read())