"""LangGraph + Gemini chat — import this module only for real LLM turns (not routine fill)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _assistant_message_content_to_str(content) -> str:
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


def _strip_message(raw: object) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw
    for _zw in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(_zw, "")
    return s.strip()


def invoke_chat_llm(message_val: str, user_id: str, raw_thread: str) -> str:
    from fastapi import HTTPException

    from app.glow_agent import (
        agent,
        ensure_no_legacy_catalog_hosts,
        sanitize_assistant_product_urls,
    )

    msg = _strip_message(message_val)
    if not msg:
        raise HTTPException(
            status_code=400,
            detail="message is required for chat (empty content is not sent to the model)",
        )
    thread_id = f"{user_id}::{raw_thread}"
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=msg)]},
        config=config,
    )
    text = _last_assistant_text(result["messages"])
    text = sanitize_assistant_product_urls(text)
    text = ensure_no_legacy_catalog_hosts(text)
    return text
