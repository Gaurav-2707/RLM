"""
FastAPI router for the RLM Engine module.

Provides session-based and stateless access to the 3-step reasoning pipeline
(Decompose → Refine → Synthesise).

Mount this router in api/main.py with:
    from api.engine_api import router as engine_router
    app.include_router(engine_router, prefix="/engine", tags=["Engine"])
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from RLM.engine_repl import EngineREPL

router = APIRouter()

# In-memory session store: session_id → EngineREPL
sessions: Dict[str, EngineREPL] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class EngineSessionRequest(BaseModel):
    model: str = "gemini-2.5-flash"

class EngineSessionResponse(BaseModel):
    session_id: str
    model: str


class EngineRunRequest(BaseModel):
    session_id: str
    problem: str

class EngineRunResponse(BaseModel):
    session_id: str
    final_output: str
    steps: List[Dict[str, Any]]


class EngineResetRequest(BaseModel):
    session_id: str

class EngineResetResponse(BaseModel):
    reset: bool


class EngineStatusResponse(BaseModel):
    session_id: str
    model: str
    steps_in_last_run: int


class ReasonRequest(BaseModel):
    """Stateless reasoning request — no session management required."""
    problem: str
    model: str = "gemini-2.5-flash"

class ReasonResponse(BaseModel):
    final_output: str
    steps: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> EngineREPL:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/session", response_model=EngineSessionResponse)
def create_session(request: EngineSessionRequest):
    """
    Create a new Engine session.

    Returns a ``session_id`` to use with subsequent ``/run`` and ``/reset`` calls.
    Sessions persist in memory for the lifetime of the server process.
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = EngineREPL(model=request.model)
    return EngineSessionResponse(session_id=session_id, model=request.model)


@router.post("/run", response_model=EngineRunResponse)
def run_engine(request: EngineRunRequest):
    """
    Run the 3-step Decompose→Refine→Synthesise reasoning pipeline.

    Returns the final synthesised answer and the full step-by-step history.
    Raises HTTP 500 if the engine throws an exception.
    """
    engine = _get_session(request.session_id)
    try:
        final_output = engine.run(request.problem)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EngineRunResponse(
        session_id=request.session_id,
        final_output=final_output,
        steps=engine.get_steps(),
    )


@router.post("/reset", response_model=EngineResetResponse)
def reset_engine(request: EngineResetRequest):
    """Reset an engine session's history without deleting the session."""
    engine = _get_session(request.session_id)
    engine.reset()
    return EngineResetResponse(reset=True)


@router.get("/status/{session_id}", response_model=EngineStatusResponse)
def get_status(session_id: str):
    """Return metadata and step count for an engine session."""
    engine = _get_session(session_id)
    return EngineStatusResponse(
        session_id=session_id,
        model=engine.model,
        steps_in_last_run=len(engine.get_steps()),
    )


@router.post("/reason", response_model=ReasonResponse)
def stateless_reason(request: ReasonRequest):
    """
    Stateless convenience endpoint — run the reasoning pipeline without
    creating or managing a session.

    Useful for one-off calls. Creates an ephemeral engine, runs it,
    and returns the result. No state is persisted.
    """
    engine = EngineREPL(model=request.model)
    try:
        final_output = engine.run(request.problem)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ReasonResponse(final_output=final_output, steps=engine.get_steps())
