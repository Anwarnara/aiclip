"""
WebSocket Handler for Real-time Updates
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.core.state import app_state

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state.websocket_clients.add(websocket)

    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "event": "connected",
            "data": {
                "status": app_state.get_status(),
                "settings": app_state.settings,
                "clips": app_state.processing.clips,
                "logs": app_state.processing.logs[-20:]  # Last 20 logs
            }
        }))

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                event = message.get("event")

                if event == "subscribe":
                    # Client is subscribing to updates (already connected)
                    await websocket.send_text(json.dumps({
                        "event": "subscribed",
                        "data": {"status": "ok"}
                    }))

                elif event == "ping":
                    await websocket.send_text(json.dumps({
                        "event": "pong",
                        "data": {}
                    }))

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        app_state.websocket_clients.discard(websocket)
