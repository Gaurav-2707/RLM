import uuid
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from RLM.acc import AdaptiveComputeController, ComplexityScorer
from RLM.acc_repl import AdaptiveRLM
from api.engine_api import router as engine_router
from api.memory_api import router as memory_router

app = FastAPI(
    title="RLM API",
    description="Unified API for the RLM Adaptive Compute Controller, Engine, and Memory modules.",
    version="1.0.0"
)

# Mount sub-routers
app.include_router(engine_router, prefix="/engine", tags=["Engine"])
app.include_router(memory_router, prefix="/memory", tags=["Memory"])

# In-memory storage for active sessions (session_id -> AdaptiveComputeController)
sessions: Dict[str, AdaptiveComputeController] = {}
scorer = ComplexityScorer()

# --- Models ---
class SessionCreateRequest(BaseModel):
    max_api_calls: Optional[int] = None

class SessionCreateResponse(BaseModel):
    session_id: str
    max_api_calls: Optional[int]

class ScoreRequest(BaseModel):
    query: str
    context: Optional[str] = None

class ScoreResponse(BaseModel):
    complexity_score: float

class QueryRequest(BaseModel):
    session_id: str
    query: str
    context: Optional[str] = None
    model: str = "ollama/llama3"
    recursive_model: str = "ollama/llama3"
    enable_logging: bool = False

class QueryResponse(BaseModel):
    response: str
    complexity_score: float
    depth_selected: int
    api_calls_used: int
    max_iterations_assigned: int

# --- Endpoints ---

@app.post("/api/v1/session", response_model=SessionCreateResponse)
def create_session(request: SessionCreateRequest):
    """Create a new ACC tracking session."""
    session_id = str(uuid.uuid4())
    acc = AdaptiveComputeController(max_api_calls=request.max_api_calls)
    acc.new_episode()
    sessions[session_id] = acc
    return SessionCreateResponse(session_id=session_id, max_api_calls=request.max_api_calls)

@app.post("/api/v1/score", response_model=ScoreResponse)
def get_score(request: ScoreRequest):
    """Pure diagnostic endpoint to compute complexity score."""
    score = scorer.score(query=request.query, context=request.context or "")
    return ScoreResponse(complexity_score=score)

@app.post("/api/v1/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """Submits a query + context, uses AdaptiveRLM to dynamically process REPL iterations."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    acc = sessions[request.session_id]
    
    adaptive_rlm = AdaptiveRLM(
        model=request.model,
        recursive_model=request.recursive_model,
        enable_logging=request.enable_logging,
        acc=acc
    )
    
    # Run the query. The depth assignment and iteration changes happen inside adaptive_rlm
    response = adaptive_rlm.completion(context=request.context or "", query=request.query)
    
    # Retrieve the last record for diagnostic return
    records = acc.records
    if not records:
        raise HTTPException(status_code=500, detail="No ACC record found for this request")
        
    last_record = records[-1]
    
    return QueryResponse(
        response=response,
        complexity_score=last_record.complexity_score,
        depth_selected=last_record.depth_selected,
        api_calls_used=acc.api_calls_used,
        max_iterations_assigned=adaptive_rlm._max_iterations
    )
