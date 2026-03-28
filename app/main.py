'''Fast API application'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.glow_agent import agent
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=request.message)]
            },
            config=config,
        )
        return ChatResponse(response=result["messages"][-1].content, 
                            thread_id=thread_id,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

