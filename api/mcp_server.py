import json
import sys
import traceback
from typing import Any, Callable, Dict, Optional

from RLM.api.context_os import context_os


def _text_result(payload: Any) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2, ensure_ascii=False),
            }
        ]
    }


def _tool_schema(name: str, description: str, properties: Dict[str, Any], required=None) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


TOOLS = [
    _tool_schema(
        "get_current_context",
        "Pack active file, terminal trace, developer intent, blast radius, and episodic memory into one local context object.",
        {
            "project_root": {"type": "string"},
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
    ),
    _tool_schema(
        "get_active_file",
        "Return the current active file, cursor, selection, and local text content if readable.",
        {},
    ),
    _tool_schema(
        "get_terminal_trace",
        "Return the latest local terminal command, exit code, and trace from RLM-Sync.",
        {},
    ),
    _tool_schema(
        "get_relevant_memories",
        "Retrieve persisted episodic memories and failure warnings relevant to the query.",
        {
            "project_root": {"type": "string"},
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
    ),
    _tool_schema(
        "get_blast_radius",
        "Return impacted dependents and upstream dependencies for a local file using the semantic graph.",
        {
            "project_root": {"type": "string"},
            "active_file": {"type": "string"},
            "max_depth": {"type": "integer", "default": 3},
            "persist_graph": {"type": "boolean", "default": False},
        },
    ),
    _tool_schema(
        "write_agent_memory",
        "Opt-in write of an agent state/action/outcome memory into the repo-local episodic memory store.",
        {
            "project_root": {"type": "string"},
            "state": {"type": "string"},
            "reasoning": {"type": "string"},
            "action": {"type": "string"},
            "outcome": {"type": "string"},
            "outcome_score": {"type": "number"},
            "parent_ids": {"type": "array", "items": {"type": "string"}},
        },
        required=["state", "action", "outcome", "outcome_score"],
    ),
]


def call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = arguments or {}
    dispatch: Dict[str, Callable[..., Any]] = {
        "get_current_context": context_os.get_current_context,
        "get_active_file": context_os.get_active_file,
        "get_terminal_trace": context_os.get_terminal_trace,
        "get_relevant_memories": context_os.get_relevant_memories,
        "get_blast_radius": context_os.get_blast_radius,
        "write_agent_memory": context_os.write_agent_memory,
    }
    if name not in dispatch:
        raise ValueError(f"Unknown MCP tool: {name}")
    return _text_result(dispatch[name](**args))


def handle_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "rlm-context-os", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = call_tool(params.get("name"), params.get("arguments") or {})
        elif method == "notifications/initialized":
            return None
        else:
            raise ValueError(f"Unsupported MCP method: {method}")

        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:
        if request_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": str(exc),
                "data": traceback.format_exc(limit=3),
            },
        }


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = handle_request(json.loads(line))
        except json.JSONDecodeError as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
