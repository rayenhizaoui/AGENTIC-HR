from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.agents.supervisor import run_supervisor

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    messages: List[Any] = []
    context: Optional[Dict[str, Any]] = {}
    reasoning_log: List[str] = []

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that delegates to the LangGraph Supervisor.
    """
    try:
        result = run_supervisor(request.message, request.context)
        return ChatResponse(
            response=result["messages"][-1].content,
            messages=result["messages"],
            context=result.get("job_context", {}),
            reasoning_log=result.get("reasoning_log", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
