import time

from RLM.api.context_os import ContextOS, load_repo_memory
from RLM.api.schemas import ExecutionState, MemoryState, SyncMessage, WorkspaceState
from RLM.api.sync_broker import broker


def _set_state(active_file: str):
    broker.current_state = SyncMessage(
        workspace=WorkspaceState(
            active_file=active_file,
            cursor_line=3,
            selected_text="login(8)",
        ),
        execution=ExecutionState(
            latest_command="pytest test_auth.py",
            exit_code=1,
            terminal_trace="AssertionError: expected session-token",
        ),
        memory=MemoryState(
            developer_intent="Fix auth boundary at password length 8",
            agent_scratchpad="",
        ),
        client_id="test",
        timestamp=time.time(),
    )


def test_context_packer_includes_sync_graph_and_memory(tmp_path):
    (tmp_path / "db.py").write_text("def save_user_session(user_id, security_level):\n    return 'token'\n")
    auth = tmp_path / "auth.py"
    auth.write_text("from db import save_user_session\n\ndef login(n):\n    return save_user_session(1, 1)\n")
    _set_state(str(auth))

    context_os = ContextOS()
    context_os.write_agent_memory(
        project_root=str(tmp_path),
        state="AssertionError password length 8 failed",
        reasoning="Boundary was strict greater-than.",
        action='{"file":"auth.py","search":"> 8","replace":">= 8"}',
        outcome="tests_passed",
        outcome_score=1.0,
    )

    packed = context_os.get_current_context(
        project_root=str(tmp_path),
        query="password length 8 failed",
    )

    assert packed["workspace"]["active_file"] == str(auth)
    assert "AssertionError" in packed["terminal_trace"]["terminal_trace"]
    assert packed["active_file"]["content"].startswith("from db import")
    assert packed["blast_radius"]["source"] == "auth.py"
    assert packed["retrieved_memories"]
    assert "RLM CONTEXT OS PACK" in packed["context_text"]


def test_agent_memory_persists_per_repo(tmp_path):
    context_os = ContextOS()
    result = context_os.write_agent_memory(
        project_root=str(tmp_path),
        state="same failing state",
        reasoning="first attempt",
        action="bad edit",
        outcome="tests_failed",
        outcome_score=-1.0,
    )

    reloaded = load_repo_memory(str(tmp_path))
    assert result["entry"]["entry_id"] == reloaded.memories[0].entry_id
    assert reloaded.memories[0].outcome == "tests_failed"
