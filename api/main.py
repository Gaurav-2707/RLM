import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from RLM.api.schemas import ExecuteRequest, ExecuteResponse, SyncMessage
from RLM.api.agent import RLMAgentWrapper
from RLM.api.sync_broker import broker
from RLM.engine.test_gen import test_generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SLINGSHOT RLM Core API",
    description="Enterprise Reasoning Engine & Autonomous Execution API",
    version="1.0.0"
)

# Allow CORS for IDE extensions and React frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RLM Cognitive Engine is running."}

@app.post("/v1/rlm/execute", response_model=ExecuteResponse)
def execute_task(request: ExecuteRequest):
    """
    Executes a task autonomously against the specified codebase using 
    Test-Driven Reinforcement Learning and Episodic Graph Memory.
    """
    logger.info(f"Received execution request for repo: {request.repo_path}")
    logger.info(f"Task: {request.task_description}")
    
    try:
        agent = RLMAgentWrapper(request)
        response = agent.execute()
        return response
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

# --- RLM-Sync Endpoints ---

@app.get("/v1/rlm-sync/state", response_model=SyncMessage)
def get_sync_state():
    """Returns the current synchronized state across all clients."""
    return broker.current_state

@app.post("/v1/rlm-sync/state", response_model=SyncMessage)
async def update_sync_state(update: SyncMessage):
    """Updates the state via REST and broadcasts to all WebSocket clients."""
    await broker.broadcast_state(update, sender=None)
    return broker.current_state

@app.post("/v1/rlm/generate-test")
def generate_auto_test():
    """
    Reads the developer's intent and active file from RLM-Sync (Memory Layer),
    and generates a failing pytest file to enable TDRL loop execution.
    """
    state = broker.current_state
    
    intent = state.memory.developer_intent
    active_file = state.workspace.active_file
    
    if not intent or not active_file:
        raise HTTPException(status_code=400, detail="Missing developer_intent or active_file in RLM-Sync memory layer.")
        
    try:
        test_code = test_generator.generate_test(active_file, intent)
        return {"status": "success", "test_file": "tests/test_auto_generated.py", "code": test_code}
    except Exception as e:
        logger.error(f"TestGen failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/v1/rlm-sync/stream")
async def sync_stream(websocket: WebSocket):
    """
    Bidirectional WebSocket stream. 
    Clients send partial state updates; broker merges and broadcasts to everyone else.
    """
    await broker.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Parse incoming message
                message = SyncMessage.model_validate_json(data)
                await broker.broadcast_state(message, sender=websocket)
            except Exception as e:
                logger.error(f"Invalid sync message received: {e}")
    except WebSocketDisconnect:
        await broker.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("RLM.api.main:app", host="127.0.0.1", port=8000, reload=True)
