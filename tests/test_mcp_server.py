import json
import time

from RLM.api.mcp_server import handle_request
from RLM.api.schemas import ExecutionState, MemoryState, SyncMessage, WorkspaceState
from RLM.api.sync_broker import broker


def test_mcp_lists_context_os_tools():
    response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    tool_names = {tool["name"] for tool in response["result"]["tools"]}
    assert {
        "get_current_context",
        "get_active_file",
        "get_terminal_trace",
        "get_relevant_memories",
        "get_blast_radius",
        "write_agent_memory",
    }.issubset(tool_names)


def test_mcp_tool_returns_current_context(tmp_path):
    active = tmp_path / "auth.py"
    active.write_text("def login():\n    return True\n")
    broker.current_state = SyncMessage(
        workspace=WorkspaceState(active_file=str(active), cursor_line=1),
        execution=ExecutionState(latest_command="pytest", exit_code=0, terminal_trace="passed"),
        memory=MemoryState(developer_intent="keep auth passing"),
        client_id="test",
        timestamp=time.time(),
    )

    response = handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_current_context",
            "arguments": {"project_root": str(tmp_path), "query": "auth"},
        },
    })

    text = response["result"]["content"][0]["text"]
    payload = json.loads(text)
    assert payload["active_file"]["active_file"] == str(active)
    assert payload["execution"]["terminal_trace"] == "passed"
