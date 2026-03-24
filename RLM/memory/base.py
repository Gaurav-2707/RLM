import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class MemoryEntry:
    """Represents a single episodic memory."""
    state: str
    reasoning: str
    action: str
    outcome: str
    outcome_score: float  # Range: -1.0 to 1.0 (or similar)
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**data)
