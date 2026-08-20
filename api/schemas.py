from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ExecuteRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute path to the target repository")
    task_description: str = Field(..., description="High-level description of the task or bug to fix")
    test_command: str = Field(..., description="The command to run tests, e.g., 'pytest test_auth.py'")
    reward_type: str = Field(default="both", description="Type of reward: 'binary', 'granular', or 'both'")
    max_steps: int = Field(default=20, description="Maximum number of RL steps to allow")

class ExecutionStep(BaseModel):
    step_number: int
    action_taken: str
    reward_received: float
    confidence: float
    context_length: float
    is_success: bool
    logs: str

class ExecuteResponse(BaseModel):
    success: bool
    final_reward: float
    total_steps: int
    conformal_safety_score: Optional[float] = Field(default=None, description="Statistical safety guarantee if available")
    trace: List[ExecutionStep]
    message: str

# --- RLM-Sync Models ---

class WorkspaceState(BaseModel):
    active_file: Optional[str] = Field(default=None, description="Absolute path of the currently open file")
    cursor_line: Optional[int] = Field(default=None, description="Line number of the user's cursor")
    selected_text: Optional[str] = Field(default=None, description="Currently highlighted text")

class ExecutionState(BaseModel):
    latest_command: Optional[str] = Field(default=None, description="The last command run in the terminal")
    exit_code: Optional[int] = Field(default=None, description="Exit code of the last command")
    terminal_trace: Optional[str] = Field(default=None, description="Recent terminal output or compiler trace")

class MemoryState(BaseModel):
    developer_intent: Optional[str] = Field(default=None, description="The developer's current high-level goal")
    agent_scratchpad: Optional[str] = Field(default=None, description="Active reasoning pad for the agent")

class SyncMessage(BaseModel):
    workspace: Optional[WorkspaceState] = Field(default_factory=WorkspaceState)
    execution: Optional[ExecutionState] = Field(default_factory=ExecutionState)
    memory: Optional[MemoryState] = Field(default_factory=MemoryState)
    client_id: str = Field(..., description="ID of the client that generated this update (e.g. 'cursor', 'terminal')")
    timestamp: float = Field(..., description="Epoch timestamp of the update")
