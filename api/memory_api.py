"""
FastAPI router for the RLM Memory module.

Provides session-based access to episodic memory (BM25 retrieval + recency decay).

Mount this router in api/main.py with:
    from api.memory_api import router as memory_router
    app.include_router(memory_router, prefix="/memory", tags=["Memory"])
"""

import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from RLM.memory_repl import MemoryREPL

router = APIRouter()

# In-memory session store: session_id → MemoryREPL
sessions: Dict[str, MemoryREPL] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MemorySessionRequest(BaseModel):
    capacity: int = 200
    decay_rate: float = 0.0001

class MemorySessionResponse(BaseModel):
    session_id: str
    capacity: int


class MemoryStoreRequest(BaseModel):
    session_id: str
    query: str
    reasoning: str
    action: str
    outcome: str
    outcome_score: float = 0.5

class MemoryStoreResponse(BaseModel):
    stored: bool
    conflicts: List[str]
    memory_count: int


class MemoryRetrieveRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 3

class MemoryRetrieveResponse(BaseModel):
    context: str          # formatted string ready to prepend to a prompt
    memory_count: int


class MemoryResetRequest(BaseModel):
    session_id: str

class MemoryResetResponse(BaseModel):
    reset: bool


class MemoryStatusResponse(BaseModel):
    session_id: str
    memory_count: int
    capacity: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> MemoryREPL:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/session", response_model=MemorySessionResponse)
def create_session(request: MemorySessionRequest):
    """
    Create a new Memory session.

    Returns a ``session_id`` used for all subsequent memory operations.
    Memory persists in-process for the lifetime of the server.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = MemoryREPL(
        capacity=request.capacity,
        decay_rate=request.decay_rate,
    )
    return MemorySessionResponse(session_id=session_id, capacity=request.capacity)


@router.post("/store", response_model=MemoryStoreResponse)
def store_memory(request: MemoryStoreRequest):
    """
    Store a new episodic memory entry.

    ``outcome_score`` should be in [-1.0, 1.0].
    Returns any conflict warnings detected (similar state, opposite outcome).
    """
    memory = _get_session(request.session_id)
    conflicts = memory.store(
        query=request.query,
        reasoning=request.reasoning,
        action=request.action,
        outcome=request.outcome,
        outcome_score=request.outcome_score,
    )
    return MemoryStoreResponse(
        stored=True,
        conflicts=conflicts,
        memory_count=memory.memory_count(),
    )


@router.post("/retrieve", response_model=MemoryRetrieveResponse)
def retrieve_memory(request: MemoryRetrieveRequest):
    """
    Retrieve the top_k most relevant memories for a given query.

    Returns a pre-formatted context string suitable for injecting into
    an LLM system prompt. Returns an empty string if no memories exist.
    """
    memory = _get_session(request.session_id)
    context = memory.retrieve_as_context(query=request.query, top_k=request.top_k)
    return MemoryRetrieveResponse(context=context, memory_count=memory.memory_count())


@router.post("/reset", response_model=MemoryResetResponse)
def reset_memory(request: MemoryResetRequest):
    """Clear all memories in a session without deleting the session."""
    memory = _get_session(request.session_id)
    memory.reset()
    return MemoryResetResponse(reset=True)


@router.get("/status/{session_id}", response_model=MemoryStatusResponse)
def get_status(session_id: str):
    """Return memory count and capacity for a session."""
    memory = _get_session(session_id)
    return MemoryStatusResponse(
        session_id=session_id,
        memory_count=memory.memory_count(),
        capacity=memory.capacity,
    )
