import json
import os
import asyncio
import logging
from typing import List
from fastapi import WebSocket
from pydantic import ValidationError
from RLM.api.schemas import SyncMessage, WorkspaceState, ExecutionState, MemoryState

logger = logging.getLogger(__name__)

class SyncBroker:
    def __init__(self, persistence_file: str = ".rlm_sync.json"):
        self.active_connections: List[WebSocket] = []
        self.persistence_file = persistence_file
        self.lock = asyncio.Lock()
        
        # Initialize default state
        self.current_state = SyncMessage(
            workspace=WorkspaceState(),
            execution=ExecutionState(),
            memory=MemoryState(),
            client_id="system",
            timestamp=0.0
        )
        self._load_persisted_state()

    def _load_persisted_state(self):
        """Loads state from local JSON file if it exists."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r") as f:
                    data = json.load(f)
                    self.current_state = SyncMessage(**data)
                logger.info(f"Loaded persisted RLM-Sync state from {self.persistence_file}")
            except Exception as e:
                logger.error(f"Failed to load persisted state: {e}")

    async def _persist_state(self):
        """Writes current state to local JSON file asynchronously."""
        try:
            with open(self.persistence_file, "w") as f:
                json.dump(self.current_state.model_dump(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
            # Send current state upon connection
            await websocket.send_text(self.current_state.model_dump_json())
        logger.info(f"Client connected. Active clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Active clients: {len(self.active_connections)}")

    async def broadcast_state(self, new_state: SyncMessage, sender: WebSocket):
        """
        Merges new state fields with the current state (partial updates),
        persists it, and broadcasts the full state to all other connected clients.
        """
        async with self.lock:
            # Merge fields (only update fields that are explicitly provided and not None)
            # This allows sparse updates from different clients.
            # Convert incoming to dict, ignoring None values
            update_data = new_state.model_dump(exclude_unset=True, exclude_none=True)
            
            # Update workspace
            if 'workspace' in update_data:
                for k, v in update_data['workspace'].items():
                    setattr(self.current_state.workspace, k, v)
            
            # Update execution
            if 'execution' in update_data:
                for k, v in update_data['execution'].items():
                    setattr(self.current_state.execution, k, v)
                    
            # Update memory
            if 'memory' in update_data:
                for k, v in update_data['memory'].items():
                    setattr(self.current_state.memory, k, v)
            
            # Update metadata
            self.current_state.client_id = new_state.client_id
            self.current_state.timestamp = new_state.timestamp

            await self._persist_state()

            # Broadcast
            message_json = self.current_state.model_dump_json()
            for connection in self.active_connections:
                if connection != sender:
                    try:
                        await connection.send_text(message_json)
                    except Exception as e:
                        logger.warning(f"Failed to send to client: {e}")

# Global instance
broker = SyncBroker()
