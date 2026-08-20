import asyncio
import websockets
import json
import time

WS_URL = "ws://127.0.0.1:8000/v1/rlm-sync/stream"

async def ide_client():
    """Simulates an IDE (like Cursor) sending user context updates."""
    print("[IDE] Connecting to RLM-Sync...")
    async with websockets.connect(WS_URL) as websocket:
        # Wait for initial state broadcast on connect
        initial_state = await websocket.recv()
        print(f"[IDE] Connected. Initial state length: {len(initial_state)}")
        
        # Simulate developer workflow
        updates = [
            {
                "client_id": "cursor_ide",
                "timestamp": time.time(),
                "workspace": {
                    "active_file": "/Users/arushsinghal/Documents/RLM/api/main.py",
                    "cursor_line": 42
                }
            },
            {
                "client_id": "cursor_ide",
                "timestamp": time.time(),
                "workspace": {
                    "cursor_line": 55,
                    "selected_text": "broker.broadcast_state(update)"
                },
                "memory": {
                    "developer_intent": "Fix the broadcasting bug in the sync broker"
                }
            }
        ]

        for i, update in enumerate(updates):
            await asyncio.sleep(2) # wait a bit before sending
            print(f"\n[IDE] Sending update {i+1}...")
            await websocket.send(json.dumps(update))
            
        # Keep connection open
        while True:
            await asyncio.sleep(1)

async def web_agent_client():
    """Simulates a Web Agent (like Claude Web UI) receiving context updates."""
    print("[WebAgent] Connecting to RLM-Sync...")
    async with websockets.connect(WS_URL) as websocket:
        print("[WebAgent] Connected. Listening for real-time context updates...")
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            client_id = data.get("client_id")
            
            # Don't print our own initialization or system messages if we don't want to, 
            # but here we print everything.
            print(f"\n[WebAgent] Received context update from '{client_id}':")
            print(json.dumps(data, indent=2))

async def main():
    # Run both clients concurrently
    await asyncio.gather(
        web_agent_client(),
        ide_client()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test finished.")
