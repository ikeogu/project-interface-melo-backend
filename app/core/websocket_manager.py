from typing import Dict, List
from fastapi import WebSocket
import json


class WebSocketManager:
    """
    Manages active WebSocket connections per user.
    Each user can have multiple connections (multiple devices).
    """

    def __init__(self):
        # user_id -> list of active websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

    async def send_to_user(self, user_id: str, message: dict):
        """Send a message to all connections for a given user."""
        if user_id not in self.active_connections:
            return
        dead = []
        for ws in self.active_connections[user_id]:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections[user_id].remove(ws)

    async def send_to_chat(self, chat_id: str, participant_ids: List[str], message: dict):
        """Broadcast to all participants in a chat."""
        for uid in participant_ids:
            await self.send_to_user(uid, message)

    def is_online(self, user_id: str) -> bool:
        return user_id in self.active_connections and len(self.active_connections[user_id]) > 0


ws_manager = WebSocketManager()
