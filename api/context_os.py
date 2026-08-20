import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from RLM.api.sync_broker import broker
from RLM.memory.base import MemoryEntry
from RLM.memory.graph import SemanticContextGraph
from RLM.memory.system import EpisodicMemorySystem


MEMORY_FILENAME = ".rlm_memory.json"
GRAPH_FILENAME = ".rlm_graph.json"


def resolve_project_root(project_root: Optional[str] = None) -> str:
    """Resolve the local repo root used by Context OS tools."""
    if project_root:
        return os.path.abspath(project_root)

    active_file = broker.current_state.workspace.active_file
    if active_file:
        path = os.path.abspath(active_file)
        return os.path.dirname(path) if os.path.isfile(path) else path

    return os.getcwd()


def memory_path_for(project_root: str) -> str:
    return os.path.join(os.path.abspath(project_root), MEMORY_FILENAME)


def graph_path_for(project_root: str) -> str:
    return os.path.join(os.path.abspath(project_root), GRAPH_FILENAME)


def load_repo_memory(project_root: str, capacity: int = 100) -> EpisodicMemorySystem:
    memory = EpisodicMemorySystem(capacity=capacity)
    memory.load(memory_path_for(project_root))
    return memory


def save_repo_memory(project_root: str, memory: EpisodicMemorySystem) -> str:
    path = memory_path_for(project_root)
    memory.save(path)
    return path


class ContextOS:
    """
    Local-first context packer for AI apps.

    This is the shared layer behind the MCP server and RLM execution loop. It
    only reads local broker/memory/graph state unless write_agent_memory is
    explicitly called.
    """

    def __init__(self, sync_broker=broker):
        self.broker = sync_broker

    def get_current_context(
        self,
        project_root: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        root = resolve_project_root(project_root)
        state = self.broker.current_state
        query_text = query or " ".join(
            part
            for part in [
                state.memory.developer_intent,
                state.execution.terminal_trace,
                state.workspace.selected_text,
                state.workspace.active_file,
            ]
            if part
        )

        active_file = self.get_active_file()
        terminal_trace = self.get_terminal_trace()
        memories = self.get_relevant_memories(root, query_text, top_k=top_k)
        blast_radius = self.get_blast_radius(root, active_file.get("active_file"))

        packed = {
            "schema_version": 1,
            "project_root": root,
            "generated_at": time.time(),
            "workspace": state.workspace.model_dump(),
            "execution": state.execution.model_dump(),
            "memory_state": state.memory.model_dump(),
            "active_file": active_file,
            "terminal_trace": terminal_trace,
            "blast_radius": blast_radius,
            "retrieved_memories": memories["retrieved_memories"],
            "failure_warnings": memories["failure_warnings"],
            "recent_actions": memories["recent_actions"],
        }
        packed["context_text"] = self._format_context_text(packed)
        packed["estimated_context_tokens"] = max(1, len(packed["context_text"]) // 4)

        # --- Context compression telemetry ---
        # tokens_served: how many tokens we're actually sending to the LLM
        tokens_served = packed["estimated_context_tokens"]

        # full_history_tokens_estimated: a conservative estimate of what the
        # caller would have had to send without Context OS (all stored memory
        # entries + full file content + complete terminal trace + workspace state).
        mem_chars = sum(
            len(m.get("state", "")) + len(m.get("action", "")) + len(m.get("outcome", ""))
            for m in packed["retrieved_memories"] + packed["recent_actions"]
        )
        file_chars = len(packed["active_file"].get("content") or "")
        trace_chars = len(packed["terminal_trace"].get("terminal_trace") or "")
        workspace_chars = sum(len(str(v)) for v in packed["workspace"].values())
        full_history_chars = mem_chars + file_chars + trace_chars + workspace_chars
        full_history_tokens = max(tokens_served, full_history_chars // 4)

        compression_ratio = round(full_history_tokens / tokens_served, 1) if tokens_served > 0 else 1.0
        # Estimated compaction calls a naive LLM wrapper would have needed
        # (Claude Code compacts at ~8k tokens; count how many times that threshold
        # would have been crossed by the full history).
        _COMPACT_THRESHOLD = 8000
        compaction_calls_avoided = max(0, full_history_tokens // _COMPACT_THRESHOLD)

        packed["tokens_served"] = tokens_served
        packed["full_history_tokens_estimated"] = full_history_tokens
        packed["compression_ratio"] = compression_ratio
        packed["compaction_calls_avoided"] = compaction_calls_avoided
        return packed

    def get_active_file(self) -> Dict[str, Any]:
        state = self.broker.current_state.workspace
        path = state.active_file
        content = None
        read_error = None
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                read_error = "active file is not utf-8 text"
            except OSError as exc:
                read_error = str(exc)

        return {
            "active_file": path,
            "cursor_line": state.cursor_line,
            "selected_text": state.selected_text,
            "content": content,
            "read_error": read_error,
        }

    def get_terminal_trace(self) -> Dict[str, Any]:
        state = self.broker.current_state.execution
        return {
            "latest_command": state.latest_command,
            "exit_code": state.exit_code,
            "terminal_trace": state.terminal_trace,
        }

    def get_relevant_memories(
        self,
        project_root: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        root = resolve_project_root(project_root)
        memory = load_repo_memory(root)
        query_text = query or self.broker.current_state.memory.developer_intent or ""
        retrieved, warnings = memory.retrieve(query_text, top_k=top_k)

        return {
            "memory_path": memory_path_for(root),
            "retrieved_memories": [
                self._memory_payload(entry, score)
                for entry, score in retrieved
            ],
            "failure_warnings": warnings,
            "recent_actions": [
                self._memory_payload(entry, None)
                for entry in sorted(memory.memories, key=lambda m: m.timestamp or 0, reverse=True)[:top_k]
            ],
        }

    def get_blast_radius(
        self,
        project_root: Optional[str] = None,
        active_file: Optional[str] = None,
        max_depth: int = 3,
        persist_graph: bool = False,
    ) -> Dict[str, Any]:
        root = resolve_project_root(project_root)
        graph = self._load_or_build_graph(root, persist_graph=persist_graph)
        source = self._node_id_for(root, active_file)
        radius = graph.get_blast_radius(source, max_depth=max_depth) if source else {}
        dependencies = graph.get_dependencies(source, max_depth=max_depth) if source else {}
        return {
            "graph_path": graph_path_for(root),
            "source": source,
            "impacted_dependents": radius,
            "upstream_dependencies": dependencies,
        }

    def write_agent_memory(
        self,
        project_root: Optional[str],
        state: str,
        action: str,
        outcome: str,
        outcome_score: float,
        reasoning: str = "",
        parent_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        root = resolve_project_root(project_root)
        memory = load_repo_memory(root)
        entry = MemoryEntry(
            state=state,
            reasoning=reasoning,
            action=action,
            outcome=outcome,
            outcome_score=outcome_score,
            parent_ids=parent_ids or [],
        )
        conflicts = memory.add_memory(entry)
        path = save_repo_memory(root, memory)
        return {
            "memory_path": path,
            "entry": entry.to_dict(),
            "conflicts": conflicts,
            "memory_count": len(memory.memories),
        }

    def _load_or_build_graph(self, project_root: str, persist_graph: bool = False) -> SemanticContextGraph:
        graph = SemanticContextGraph()
        path = graph_path_for(project_root)
        if os.path.exists(path):
            try:
                graph.load_from_disk(path)
                return graph
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass

        graph.build_from_directory(project_root)
        if persist_graph:
            try:
                graph.save_to_disk(path)
            except OSError:
                pass
        return graph

    def _node_id_for(self, project_root: str, active_file: Optional[str]) -> Optional[str]:
        if not active_file:
            return None
        path = os.path.abspath(active_file)
        try:
            return os.path.relpath(path, project_root)
        except ValueError:
            return active_file

    def _memory_payload(self, entry: MemoryEntry, score: Optional[float]) -> Dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "state": entry.state,
            "reasoning": entry.reasoning,
            "action": entry.action,
            "outcome": entry.outcome,
            "outcome_score": entry.outcome_score,
            "timestamp": entry.timestamp,
            "score": score,
            "parent_ids": entry.parent_ids,
        }

    def _format_context_text(self, packed: Dict[str, Any]) -> str:
        workspace = packed["workspace"]
        execution = packed["execution"]
        memory_state = packed["memory_state"]
        active = packed["active_file"]

        memory_lines = [
            f"- {m['action']} -> {m['outcome']} (score={m.get('score')})"
            for m in packed["retrieved_memories"]
        ] or ["- none"]
        warning_lines = packed["failure_warnings"] or ["none"]
        radius_lines = [
            f"- {node} depth={attrs.get('depth')} type={attrs.get('type')}"
            for node, attrs in packed["blast_radius"]["impacted_dependents"].items()
        ] or ["- none"]

        selected_or_content = active.get("selected_text") or active.get("content") or ""
        selected_or_content = selected_or_content[:6000]

        return "\n".join([
            "RLM CONTEXT OS PACK",
            f"Active file: {workspace.get('active_file')}",
            f"Cursor line: {workspace.get('cursor_line')}",
            f"Developer intent: {memory_state.get('developer_intent')}",
            f"Latest command: {execution.get('latest_command')}",
            f"Exit code: {execution.get('exit_code')}",
            "Terminal trace:",
            (execution.get("terminal_trace") or "")[-4000:],
            "Impacted dependents:",
            *radius_lines,
            "Episodic failure warnings:",
            *warning_lines,
            "Relevant memories:",
            *memory_lines,
            "Active selection/file excerpt:",
            selected_or_content,
        ])


context_os = ContextOS()
