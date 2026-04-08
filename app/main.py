'''Fast API application'''

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.glow_agent import agent
from app.auth import get_current_user, get_user_id 
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uuid
import os
from starlette.requests import Request

load_dotenv()

app = FastAPI(
    title="Glow Agent API",
    description="A skincare assistant powered by LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/")
async def root():
    return RedirectResponse(url="/ui")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ── Protected /me endpoint — returns the logged-in user's info ──
@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """
    Returns the Auth0 user claims from the JWT.
    Useful for the frontend to confirm who is logged in.
    """
    return {
        "sub":   user.get("sub"),
        "email": user.get("email"),
        "name":  user.get("name"),
    }
 
 # ── Logout endpoint ──
@app.get("/logout")
async def logout(request: Request):
    auth0_domain = os.getenv("AUTH0_DOMAIN")
    client_id = os.getenv("AUTH0_CLIENT_ID")
    return_to = "http://localhost:8000/ui"
    
    return RedirectResponse(
        url=f"https://{auth0_domain}/v2/logout?client_id={client_id}&returnTo={return_to}&federated"
    )

# ── Protected /chat endpoint ──
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_user_id),   # <-- requires valid JWT
):
    """
    Processes a chat message. Uses user_id (from Auth0 JWT sub claim)
    as part of the thread_id so each user has their own conversation history.
    """
    try:
        # Prefix thread_id with user_id so threads are user-scoped
        # (prevents one user from accessing another's conversation)
        raw_thread = request.thread_id or str(uuid.uuid4())
        thread_id  = f"{user_id}::{raw_thread}"
 
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
 
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
 
        return ChatResponse(
            response=result["messages"][-1].content,
            thread_id=raw_thread,    # send back the un-prefixed version to the client
        )
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
 