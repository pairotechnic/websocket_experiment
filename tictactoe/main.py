# Standard Library Imports

# Third-Party Library Imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Local Application Imports
from manager import GameManager


app = FastAPI()
manager = GameManager()

@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/{player_id}")
async def websocket_endpoint(ws: WebSocket, player_id: str):
    game_id = None
    player_index = None

    try:
        result = await manager.connect(player_id, ws)
        if result : 
            game_id, player_index = result

        # Main message loop
        async for data in ws.iter_json():
            if data.get("type") == "move":
                if game_id and player_index is not None:
                    await manager.handle_move(game_id, player_index, data["cell"])

    except WebSocketDisconnect:
        if game_id and player_index is not None:
            await manager.disconnect(game_id, player_index)