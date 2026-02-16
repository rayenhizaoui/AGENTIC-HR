from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.api import router as api_router
from app.agents.supervisor import run_supervisor
from app.agents.shared.state import AgentState
from app.agents.supervisor import (
    understand_user_message, normalize_prompt,
    supervisor_graph, FINISH
)
from langchain_core.messages import HumanMessage
import uvicorn
import json
import os
import asyncio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Intelligent HR Recruitment API",
    description="Backend API for the HR Recruitment Platform powered by Multi-Agent System",
    version="2.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:5173",  # React Frontend (Vite)
    "http://localhost:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router — no prefix so frontend calls /chat, /search, etc. directly
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Intelligent HR Recruitment API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ── WebSocket endpoint for real-time chat ─────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streamed reasoning logs.
    
    Sends incremental log steps BEFORE the final response so the frontend
    can display each step appearing in real-time.
    
    Message types:
      {"type": "log",      "step": "📥 Input received: ..."}
      {"type": "response", "response": "...", "context": {...}, "reasoning_log": [...]}
    """
    await websocket.accept()
    job_context: dict = {}

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            user_message = payload.get("message", "")
            job_context = payload.get("context", job_context)

            if not user_message.strip():
                await websocket.send_json({
                    "type": "response",
                    "response": "Please provide a message.",
                    "context": job_context,
                    "reasoning_log": [],
                })
                continue

            try:
                # ── Phase 1: LLM-powered understanding with streamed reasoning ──
                await websocket.send_json({
                    "type": "log",
                    "step": f"📥 Input: \"{user_message[:100]}{'...' if len(user_message) > 100 else ''}\""
                })
                await asyncio.sleep(0.05)

                # Normalize (fix typos, expand abbreviations)
                normalized = normalize_prompt(user_message)
                if normalized.lower().strip() != user_message.lower().strip():
                    await websocket.send_json({
                        "type": "log",
                        "step": f"📝 Normalized: \"{normalized[:100]}\""
                    })
                    await asyncio.sleep(0.05)

                # Unified LLM NLU understanding
                understanding = await asyncio.to_thread(understand_user_message, user_message)

                engine = "🤖 Mistral LLM" if understanding.get("llm_used") else "📋 Rule-based"
                await websocket.send_json({
                    "type": "log",
                    "step": f"🧠 Intent: **{understanding['intent']}** ({understanding['confidence']:.0%}) [{engine}]"
                })
                await asyncio.sleep(0.05)

                if understanding.get("sub_action"):
                    await websocket.send_json({
                        "type": "log",
                        "step": f"🎯 Sub-action: {understanding['sub_action']}"
                    })
                    await asyncio.sleep(0.05)

                entities = understanding.get("entities", {})
                if entities.get("role"):
                    await websocket.send_json({
                        "type": "log",
                        "step": f"👤 Role: {entities['role']}"
                    })
                    await asyncio.sleep(0.05)

                if entities.get("skills") and isinstance(entities["skills"], list):
                    await websocket.send_json({
                        "type": "log",
                        "step": f"🔧 Skills: {', '.join(str(s) for s in entities['skills'][:6])}"
                    })
                    await asyncio.sleep(0.05)

                if entities.get("location"):
                    await websocket.send_json({
                        "type": "log",
                        "step": f"📍 Location: {entities['location']}"
                    })
                    await asyncio.sleep(0.05)

                if entities.get("level"):
                    await websocket.send_json({
                        "type": "log",
                        "step": f"📊 Level: {entities['level']}"
                    })
                    await asyncio.sleep(0.05)

                lang = understanding.get("language", "EN")
                await websocket.send_json({
                    "type": "log",
                    "step": f"🌍 Language: {lang}"
                })
                await asyncio.sleep(0.05)

                rephrased = understanding.get("rephrased_query", "")
                if rephrased and rephrased.lower().strip() != user_message.lower().strip():
                    await websocket.send_json({
                        "type": "log",
                        "step": f"💬 Understood as: \"{rephrased[:100]}\""
                    })
                    await asyncio.sleep(0.05)

                route = understanding.get("route", "Lead_Recruiter")
                await websocket.send_json({
                    "type": "log",
                    "step": f"🔀 Route → **{route}**"
                })
                await asyncio.sleep(0.05)

                if understanding.get("reasoning"):
                    await websocket.send_json({
                        "type": "log",
                        "step": f"💡 {understanding['reasoning']}"
                    })
                    await asyncio.sleep(0.05)

                await websocket.send_json({
                    "type": "log",
                    "step": f"🚀 Executing **{route}** agent..."
                })
                await asyncio.sleep(0.05)

                # ── Phase 2: Run the full graph ──
                result = await asyncio.to_thread(run_supervisor, user_message, job_context)
                response_text = result["messages"][-1].content
                job_context = result.get("job_context", job_context)

                await websocket.send_json({
                    "type": "log",
                    "step": "✅ Done"
                })
                await asyncio.sleep(0.05)

                # ── Phase 3: Send final response ──
                await websocket.send_json({
                    "type": "response",
                    "response": response_text,
                    "context": job_context,
                    "reasoning_log": result.get("reasoning_log", []),
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "response",
                    "response": f"❌ Error: {str(e)}",
                    "context": job_context,
                    "reasoning_log": [],
                })

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
