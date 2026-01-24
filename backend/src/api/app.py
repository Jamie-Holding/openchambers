import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain/Graph imports
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

# Local imports
from backend.config.settings import DATABASE_URL
from backend.src.chatbot.agent import ask_agent, create_hansard_agent

logger = logging.getLogger(__name__)

# --- Database Configuration ---
# Standardize URI for psycopg compatibility
DATABASE_URL = DATABASE_URL.replace("+psycopg2", "").replace("+psycopg", "")

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the database connection pool and agent initialization.
    The 'yield' must stay inside the 'async with' block to keep the 
    connections open while the app is running.
    """
    async with AsyncConnectionPool(
            conninfo=DATABASE_URL,
            max_size=5,
            kwargs={"autocommit": True, "row_factory": dict_row}
    ) as pool:
        # Initialize the LangGraph checkpointer
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # Compile the graph and store it in app.state for access in routes
        app.state.graph = create_hansard_agent(checkpointer)
        app.state.pool = pool

        logger.info("Backend services started: Database pool and Agent initialized.")
        
        yield  # <--- Application handles requests here
        
    logger.info("Backend services shut down: Database pool closed.")

# --- App Initialization ---
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str

# --- Routes ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    """
    Endpoint to stream agent responses. 
    Accesses the pre-compiled graph from the application state.
    """
    # Use request.app.state to ensure we get the objects from lifespan
    graph = request.app.state.graph

    async def event_generator():
        try:
            async for token in ask_agent(
                graph,
                req.thread_id,
                req.message
            ):
                yield token
        except Exception:
            logger.exception("Streaming chat failed")
            yield "event: error\ndata: Internal error\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )