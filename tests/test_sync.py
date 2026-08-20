import pytest
from fastapi.testclient import TestClient
import os
import json
import asyncio
from RLM.api.main import app
from RLM.api.sync_broker import broker

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_persistence():
    """Ensure the persistence file is clean before and after each test."""
    if os.path.exists(".rlm_sync.json"):
        os.remove(".rlm_sync.json")
    # Reset broker state
    from RLM.api.schemas import SyncMessage, WorkspaceState, ExecutionState, MemoryState
    broker.current_state = SyncMessage(
        workspace=WorkspaceState(),
        execution=ExecutionState(),
        memory=MemoryState(),
        client_id="system",
        timestamp=0.0
    )
    broker.active_connections = []
    yield
    if os.path.exists(".rlm_sync.json"):
        os.remove(".rlm_sync.json")

def test_get_initial_state():
    response = client.get("/v1/rlm-sync/state")
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == "system"

def test_post_update_state():
    update_payload = {
        "client_id": "pytest",
        "timestamp": 123456.78,
        "workspace": {
            "active_file": "/fake/path.py",
            "cursor_line": 10
        }
    }
    response = client.post("/v1/rlm-sync/state", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == "pytest"
    assert data["workspace"]["active_file"] == "/fake/path.py"
    
    # Verify persistence file was created
    assert os.path.exists(".rlm_sync.json")
    with open(".rlm_sync.json", "r") as f:
        saved = json.load(f)
        assert saved["workspace"]["active_file"] == "/fake/path.py"

def test_partial_update_merging():
    """Ensure that updating execution state doesn't wipe workspace state."""
    # First update workspace
    client.post("/v1/rlm-sync/state", json={
        "client_id": "pytest",
        "timestamp": 1.0,
        "workspace": {"active_file": "/merge.py"}
    })
    
    # Next update execution only
    client.post("/v1/rlm-sync/state", json={
        "client_id": "pytest2",
        "timestamp": 2.0,
        "execution": {"latest_command": "ls"}
    })
    
    response = client.get("/v1/rlm-sync/state")
    data = response.json()
    
    # Both should exist
    assert data["workspace"]["active_file"] == "/merge.py"
    assert data["execution"]["latest_command"] == "ls"

def test_websocket_broadcast():
    """Test that connected WS clients receive broker broadcasts."""
    with client.websocket_connect("/v1/rlm-sync/stream") as ws:
        # Drain initial state broadcast.
        ws.receive_json()

        update_payload = {
            "client_id": "rest",
            "timestamp": 3.0,
            "memory": {"developer_intent": "websocket test"}
        }
        response = client.post("/v1/rlm-sync/state", json=update_payload)
        assert response.status_code == 200

        received = ws.receive_json()
        assert received["client_id"] == "rest"
        assert received["memory"]["developer_intent"] == "websocket test"
