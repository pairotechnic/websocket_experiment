# Standard Library Imports
import asyncio
from pathlib import Path

# Third-Party Library Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Local Application Imports
from manager import GameManager, AIGameManager
from ai.minimax import MinimaxAgent
from ai.q_agent import QAgent


app = FastAPI()
manager = GameManager()

# -----------------------------------------------------------------
# Q-agent - load the latest .pkl file from ai/models/ at startup
# -----------------------------------------------------------------

def _load_latest_q_agent() -> QAgent | None:
    models_dir = Path("ai/models")
    if not models_dir.exists():
        return None
    pkls = sorted(models_dir.glob("*.pkl"))
    if not pkls :
        return None
    agent = QAgent()
    agent.load(pkls[-1])    # alphabetical order: timestamp suffix means latest = last
    agent.epsilon = 0.0     # fully greedy - no exploration during play
    return agent

q_agent = _load_latest_q_agent()

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
    
@app.get("/tictactoe/q-agent")
async def get_q_agent_game():
    with open("game.html") as f:
        return HTMLResponse(f.read())

# ENDPOINTS - websockets
@app.websocket("/tictactoe/ws/minimax/{player_id}")
async def minimax_ws(ws: WebSocket, player_id: str, symbol: str = "X"):
    """Human vs Minimax - Symbol passed as query param ?symbol=X or ?symbol=O."""
    agent = MinimaxAgent()
    mgr = AIGameManager(agent)
    await mgr.run(player_id, ws, human_symbol=symbol)

@app.websocket("/tictactoe/ws/q-agent/{player_id}")
async def q_agent_ws(ws: WebSocket, player_id: str, symbol: str = "X"):
    """Human vs self-trained Q-agent"""
    if q_agent is None:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "No trained model found. Run ai/train.py first."})
        await ws.close()
        return
    mgr = AIGameManager(q_agent)
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