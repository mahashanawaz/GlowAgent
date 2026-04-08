'''Fast API application'''

import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
from starlette.requests import Request

from app.auth import get_current_user, get_user_id
from app.glow_agent import agent, ensure_no_legacy_catalog_hosts, sanitize_assistant_product_urls
from app.routine_recommend import recommend_routine_from_profile

load_dotenv()

app = FastAPI(
    title="Glow Agent API",
    description="A skincare assistant powered by LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _assistant_message_content_to_str(content) -> str:
    """Normalize LangChain / provider message content to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if content and all(isinstance(b, str) for b in content):
            return "".join(content)
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = (block.get("type") or "").lower()
                if btype in (
                    "thinking",
                    "executable_code",
                    "code_execution_result",
                    "tool_use",
                    "function_call",
                    "image_url",
                ):
                    continue
                if btype not in ("", "text") and "text" not in block and "content" not in block:
                    continue
                t = block.get("text")
                if t is None:
                    t = block.get("content")
                if isinstance(t, str) and t:
                    parts.append(t)
            else:
                t = getattr(block, "text", None)
                if isinstance(t, str) and t:
                    parts.append(t)
        if parts:
            return "".join(parts)
        return str(content)
    if isinstance(content, dict):
        t = content.get("text") or content.get("content")
        if isinstance(t, str):
            return t
    return str(content) if content is not None else ""


def _message_type(msg: object) -> str:
    if isinstance(msg, AIMessage):
        return "ai"
    if isinstance(msg, HumanMessage):
        return "human"
    if isinstance(msg, ToolMessage):
        return "tool"
    if isinstance(msg, BaseMessage):
        return (getattr(msg, "type", None) or "").lower()
    if isinstance(msg, dict):
        t = (msg.get("type") or msg.get("role") or "").lower()
        if t in ("human", "user"):
            return "human"
        if t in ("ai", "assistant"):
            return "ai"
        if t == "tool":
            return "tool"
    return ""


def _message_content(msg: object):
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


def _last_assistant_text(messages) -> str:
    """
    Use AI text from the current turn only (after the latest human message), and
    prefer the longest chunk so a brief trailing AI line does not replace the
    main answer. Handles dict-shaped messages from some checkpoint formats.
    """
    if not messages:
        return ""
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if _message_type(messages[i]) == "human":
            last_human_idx = i
            break
    turn_ai: list[str] = []
    for msg in messages[last_human_idx + 1 :]:
        if _message_type(msg) == "ai":
            t = _assistant_message_content_to_str(_message_content(msg)).strip()
            if t:
                turn_ai.append(t)
    if turn_ai:
        return max(turn_ai, key=len)
    for msg in reversed(messages):
        if _message_type(msg) == "ai":
            t = _assistant_message_content_to_str(_message_content(msg)).strip()
            if t:
                return t
    return _assistant_message_content_to_str(_message_content(messages[-1]))


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class RoutineRecommendRequest(BaseModel):
    skin_type: str = ""
    concerns: List[str] = []
    allergies: List[str] = []


class RoutineSlotResponse(BaseModel):
    step: str
    product: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[str] = None
    link: Optional[str] = None
    image: Optional[str] = None


class RoutineRecommendResponse(BaseModel):
    am: List[RoutineSlotResponse]
    pm: List[RoutineSlotResponse]


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Returns Auth0 user claims from the JWT."""
    return {
        "sub": user.get("sub"),
        "email": user.get("email"),
        "name": user.get("name"),
    }


@app.get("/logout")
async def logout(request: Request):
    auth0_domain = os.getenv("AUTH0_DOMAIN")
    client_id = os.getenv("AUTH0_CLIENT_ID")
    return_to = "http://localhost:8000/ui"

    return RedirectResponse(
        url=f"https://{auth0_domain}/v2/logout?client_id={client_id}&returnTo={return_to}&federated"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Processes a chat message. thread_id is prefixed with user_id so each user has
    their own conversation history.
    """
    try:
        raw_thread = request.thread_id or str(uuid.uuid4())
        thread_id = f"{user_id}::{raw_thread}"

        config = {"configurable": {"thread_id": thread_id}}

        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

        text = _last_assistant_text(result["messages"])
        text = sanitize_assistant_product_urls(text)
        text = ensure_no_legacy_catalog_hosts(text)

        return ChatResponse(
            response=text,
            thread_id=raw_thread,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/routine/recommend", response_model=RoutineRecommendResponse)
async def routine_recommend(
    body: RoutineRecommendRequest,
    _user_id: str = Depends(get_user_id),
):
    """
    Top database pick per AM/PM routine step from skin type, concerns, and allergies.
    """
    try:
        out = recommend_routine_from_profile(
            skin_type=body.skin_type,
            concerns=body.concerns,
            allergies=body.allergies,
        )
        return RoutineRecommendResponse(
            am=[RoutineSlotResponse(**s) for s in out["am"]],
            pm=[RoutineSlotResponse(**s) for s in out["pm"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
