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
        await manager.connect(player_id, ws)

        # Figure out which game this socket ended up in
        for gid, game in manager.games.items():
            if ws in game.players:
                game_id = gid
                player_index = game.players.index(ws)
                break

        # Main message loop
        async for data in ws.iter_json():
            msg_type = data.get("type")

            if msg_type == "move":
                # Refresh game_id/index in case we just got matched
                if game_id is None:
                    for gid, game in manager.games.items():
                        if ws in game.players:
                            game_id = gid
                            player_index = game.players.index(ws)
                            break
                if game_id and player_index is not None:
                    await manager.handle_move(game_id, player_index, data["cell"])

    except WebSocketDisconnect:
        if game_id and player_index is not None:
            await manager.disconnect(game_id, player_index)