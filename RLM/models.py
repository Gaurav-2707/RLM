from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import time

class MemoryEntry(BaseModel):
    """
    Represents a single episodic memory entry. 
    Compatible with the EpisodicMemorySystem.
    """
    state: str
    reasoning: str
    action: str
    outcome: str
    outcome_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class AgentState(BaseModel):
    """
    Tracks the internal state of the AgentController.
    """
    session_id: str
    current_iteration: int = 0
    max_iterations: int = 10
    task_description: str
    current_context: str = ""
    accumulated_reasoning: List[str] = []
    status: str = "initializing" # initializing, searching_memory, reasoning, acting, updating_memory, completed, failed
    is_active: bool = True

class ReasoningOutput(BaseModel):
    """
    Structured response from the Recursive Reasoning module.
    """
    rationale: str
    proposed_action: str
    action_parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_terminal: bool = False

class ActionResult(BaseModel):
    """
    Outcome of an action executed by the controller.
    """
    observation: str
    success: bool
    error_message: Optional[str] = None
    outcome_score: float = 0.0 # Used for memory weighting
    metadata: Dict[str, Any] = Field(default_factory=dict)
