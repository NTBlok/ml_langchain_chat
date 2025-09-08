from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
import logging

# Import the LLM functionality
from llm import get_llm_response, session_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Chat API",
    description="API for ML-powered chat with data integration",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for maintaining conversation context. If not provided, a new session will be created."
    )

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    error: Optional[bool] = False

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Process a chat message and return the AI's response.
    Maintains conversation context using the provided session_id.
    """
    try:
        # Generate a new session ID if none provided
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Get the response from the LLM
        response = get_llm_response(
            user_input=chat_request.message,
            session_id=session_id
        )
        
        return {
            "response": response["content"],
            "session_id": session_id,
            "timestamp": response["timestamp"],
            "error": response.get("error", False)
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your request: {str(e)}"
        )

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific chat session."""
    try:
        session = session_manager.get_session(session_id)
        return {
            "session_id": session_id,
            "created_at": getattr(session, 'created_at', 'Not available'),
            "message_count": len(session.memory.chat_history.messages) if hasattr(session.memory, 'chat_history') else 0
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "ml-chat-api",
        "version": "0.1.0"
    }
