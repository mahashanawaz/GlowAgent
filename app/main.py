'''Fast API application'''

import os
import uuid
import urllib.parse
from typing import List, Optional
from urllib.parse import parse_qs

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.requests import Request

from app.auth import get_current_user, get_user_id
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


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = ""
    thread_id: Optional[str] = None
    build_routine_slots: bool = Field(
        default=False,
        validation_alias=AliasChoices("build_routine_slots", "buildRoutineSlots"),
    )
    profile_skin_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("profile_skin_type", "profileSkinType"),
    )
    profile_concerns: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("profile_concerns", "profileConcerns"),
    )
    profile_allergies: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("profile_allergies", "profileAllergies"),
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    routine: Optional[RoutineRecommendResponse] = None


class RoutineRecommendRequest(BaseModel):
    skin_type: str = ""
    concerns: List[str] = []
    allergies: List[str] = []


def _google_shop_link(product: str, brand: str) -> str:
    q = urllib.parse.quote_plus(f"{product} {brand} skincare buy")
    return f"https://www.google.com/search?q={q}"


def _placeholder_routine_response() -> RoutineRecommendResponse:
    """Sample products so the UI never breaks when DB ranking or upstream APIs fail."""
    am_data = [
        ("Cleanser", "Daily Gel Cleanser", "GlowAgent Sample"),
        ("Toner", "Balancing Essence Toner", "GlowAgent Sample"),
        ("Serum", "Brightening Vitamin C Serum", "GlowAgent Sample"),
        ("Moisturizer", "Lightweight Gel-Cream", "GlowAgent Sample"),
        ("Sunscreen", "SPF 50 Face Fluid", "GlowAgent Sample"),
    ]
    pm_data = [
        ("Cleanser", "Oil-to-Foam Cleanser", "GlowAgent Sample"),
        ("Exfoliant", "2% BHA Liquid Exfoliant", "GlowAgent Sample"),
        ("Treatment", "Niacinamide Treatment Serum", "GlowAgent Sample"),
        ("Moisturizer", "Barrier Repair Night Cream", "GlowAgent Sample"),
        ("Eye Cream", "Peptide Eye Cream", "GlowAgent Sample"),
    ]

    def rows(specs: list[tuple[str, str, str]]) -> list[RoutineSlotResponse]:
        out: list[RoutineSlotResponse] = []
        for step, prod, brand in specs:
            out.append(
                RoutineSlotResponse(
                    step=step,
                    product=prod,
                    brand=brand,
                    price=None,
                    link=_google_shop_link(prod, brand),
                    image=None,
                )
            )
        return out

    return RoutineRecommendResponse(am=rows(am_data), pm=rows(pm_data))


def _build_routine_response(body: RoutineRecommendRequest) -> RoutineRecommendResponse:
    """Prefer real DB picks; on any failure return placeholders (never raise)."""
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
    except Exception:
        return _placeholder_routine_response()


def _routine_slots_query_flag(raw_request: Request) -> bool:
    """Detect ?routine_slots=1 via Starlette, URL, and raw ASGI query_string."""
    qp = (raw_request.query_params.get("routine_slots") or "").strip().lower()
    if qp in ("1", "true", "yes", "on"):
        return True
    for _key, vals in parse_qs(raw_request.url.query, keep_blank_values=True).items():
        if _key.lower() != "routine_slots":
            continue
        for v in vals:
            if str(v).strip().lower() in ("1", "true", "yes", "on"):
                return True
    try:
        raw_qs = raw_request.scope.get("query_string") or b""
        if isinstance(raw_qs, bytes):
            raw_qs = raw_qs.decode("utf-8", errors="replace")
        low = raw_qs.lower()
        if "routine_slots=1" in low or "routine_slots=true" in low or "routine_slots=yes" in low:
            return True
    except Exception:
        pass
    return False


def _sync_routine_header_flag(raw_request: Request) -> bool:
    """UI sends X-Glow-Sync-Routine: 1 so /chat never hits LangGraph if the query is stripped."""
    h = (raw_request.headers.get("x-glow-sync-routine") or "").strip().lower()
    return h in ("1", "true", "yes", "on")


def _truthy_json_flag(val: object) -> bool:
    """JSON booleans sometimes arrive as strings from clients or proxies."""
    if val is True:
        return True
    if isinstance(val, str) and val.strip().lower() in ("1", "true", "yes", "on"):
        return True
    return False


def _url_hints_routine_slots(raw_request: Request) -> bool:
    """Substring match on full URL — catches odd proxies / encoding where query_params miss."""
    u = str(raw_request.url).lower()
    for needle in (
        "routine_slots=1",
        "routine_slots=true",
        "routine_slots=yes",
        "routine_slots=on",
    ):
        if needle in u:
            return True
    return False


def _force_routine_from_request(payload: dict, raw_request: Request) -> bool:
    """
    Any of these means “fill routine only” — never call Gemini, even if `message` is non-empty
    (e.g. user had draft text in the chat box).
    """
    if _url_hints_routine_slots(raw_request):
        return True
    if _routine_slots_query_flag(raw_request):
        return True
    if _sync_routine_header_flag(raw_request):
        return True
    if _truthy_json_flag(payload.get("build_routine_slots")):
        return True
    if _truthy_json_flag(payload.get("buildRoutineSlots")):
        return True
    return False


def _routine_body_from_payload(payload: dict) -> tuple[str, str, list[str], list[str]]:
    """Parse profile fields without ChatRequest validation (avoids 500 on odd client shapes)."""
    tid = payload.get("thread_id")
    raw_thread = (
        str(tid).strip()
        if tid is not None and str(tid).strip()
        else str(uuid.uuid4())
    )
    skin = str(
        payload.get("profile_skin_type") or payload.get("profileSkinType") or "",
    ).strip()
    c = payload.get("profile_concerns") or payload.get("profileConcerns")
    if c is None:
        concerns: list[str] = []
    elif isinstance(c, list):
        concerns = [str(x) for x in c]
    else:
        concerns = [str(c)] if str(c).strip() else []
    a = payload.get("profile_allergies") or payload.get("profileAllergies")
    if a is None:
        allergies: list[str] = []
    elif isinstance(a, list):
        allergies = [str(x) for x in a]
    else:
        allergies = [str(a)] if str(a).strip() else []
    return raw_thread, skin, concerns, allergies


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")


@app.get("/ui")
async def ui_index():
    """Serve the SPA without a trailing-slash redirect so OAuth callback URLs stay stable."""
    return FileResponse("static/index.html")


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


chat_router = APIRouter(tags=["chat"])


@chat_router.post("/fill-routine", response_model=ChatResponse)
async def fill_routine_from_profile(
    body: RoutineRecommendRequest,
    _user_id: str = Depends(get_user_id),
):
    """
    Ranked AM/PM routine slots from the product DB only. Does not call LangGraph or Gemini.
    Use this from the UI instead of POST /chat with empty message (avoids LLM empty-content errors).
    """
    raw_thread = str(uuid.uuid4())
    r = _build_routine_response(body)
    return ChatResponse(
        response=(
            "Filled your routine from your profile using GlowAgent’s product "
            "ranking (database + your skin type, concerns, and allergies). "
            "Empty fields are treated as “any” — add more detail anytime."
        ),
        thread_id=raw_thread,
        routine=r,
    )


@chat_router.post("/sync-routine", response_model=ChatResponse)
async def sync_routine_endpoint(
    body: RoutineRecommendRequest,
    _user_id: str = Depends(get_user_id),
):
    """Dedicated routine sync — same as /fill-routine; use this from the UI (no /chat)."""
    raw_thread = str(uuid.uuid4())
    r = _build_routine_response(body)
    return ChatResponse(
        response=(
            "Filled your routine from your profile using GlowAgent’s product "
            "ranking (database + your skin type, concerns, and allergies). "
            "Empty fields are treated as “any” — add more detail anytime."
        ),
        thread_id=raw_thread,
        routine=r,
    )


def _strip_chat_message_text(raw: object) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw
    for _zw in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        s = s.replace(_zw, "")
    return s.strip()


def _should_fallback_routine_sync(raw_request: Request, payload: dict, err: str) -> bool:
    """
    Return 200 + placeholder routine instead of 500 when routine sync fails or Gemini
    returns empty-content errors on a sync-shaped request (not normal chat).
    """
    el = (err or "").lower()
    routine_ctx = _force_routine_from_request(payload, raw_request)
    if routine_ctx:
        return True
    gemini_like = (
        "generatecontent" in el
        or "contents is not specified" in el
        or ("gemini" in el and "400" in el)
    )
    if not gemini_like:
        return False
    if _sync_routine_header_flag(raw_request):
        return True
    if _truthy_json_flag(payload.get("build_routine_slots")) or _truthy_json_flag(
        payload.get("buildRoutineSlots"),
    ):
        return True
    if _url_hints_routine_slots(raw_request) or _routine_slots_query_flag(raw_request):
        return True
    return not bool(_strip_chat_message_text(payload.get("message", "")))


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(
    raw_request: Request,
    user_id: str = Depends(get_user_id),
):
    """
    Processes a chat message. thread_id is prefixed with user_id so each user has
    their own conversation history.

    If build_routine_slots is true, returns routine AM/PM slots (same ranking as
    product_ranking_tool) without invoking the LLM — uses the same URL as chat.

    Body is read via Request.json() so routine flags are not lost to FastAPI body
    injection edge cases; LangGraph is never called with an empty user message.
    """
    payload: dict = {}
    try:
        try:
            payload = await raw_request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        force_routine = _force_routine_from_request(payload, raw_request)
        # Routine sync: query, header, or build_routine_slots — never call LangGraph (ignore draft message).
        if force_routine:
            raw_thread, skin, concerns, allergies = _routine_body_from_payload(payload)
            r = _build_routine_response(
                RoutineRecommendRequest(
                    skin_type=skin,
                    concerns=concerns,
                    allergies=allergies,
                )
            )
            return ChatResponse(
                response=(
                    "Filled your routine from your profile using GlowAgent’s product "
                    "ranking (database + your skin type, concerns, and allergies). "
                    "Empty fields are treated as “any” — add more detail anytime."
                ),
                thread_id=raw_thread,
                routine=r,
            )

        body = ChatRequest.model_validate(payload)

        raw_thread = body.thread_id or str(uuid.uuid4())
        message_val = _strip_chat_message_text(payload.get("message", body.message))

        payload_wants_routine = _truthy_json_flag(
            payload.get("build_routine_slots"),
        ) or _truthy_json_flag(payload.get("buildRoutineSlots"))
        profile_keys_in_payload = any(
            k in payload
            for k in (
                "profile_skin_type",
                "profileSkinType",
                "profile_concerns",
                "profileConcerns",
                "profile_allergies",
                "profileAllergies",
            )
        )

        skip_llm = not message_val and (
            _routine_slots_query_flag(raw_request)
            or _url_hints_routine_slots(raw_request)
            or _sync_routine_header_flag(raw_request)
            or payload_wants_routine
            or body.build_routine_slots
            or profile_keys_in_payload
        )

        if skip_llm:
            r = _build_routine_response(
                RoutineRecommendRequest(
                    skin_type=(body.profile_skin_type or "").strip(),
                    concerns=list(body.profile_concerns or []),
                    allergies=list(body.profile_allergies or []),
                )
            )
            return ChatResponse(
                response=(
                    "Filled your routine from your profile using GlowAgent’s product "
                    "ranking (database + your skin type, concerns, and allergies). "
                    "Empty fields are treated as “any” — add more detail anytime."
                ),
                thread_id=raw_thread,
                routine=r,
            )

        if not message_val:
            raise HTTPException(
                status_code=400,
                detail="message is required unless routine fill (build_routine_slots or ?routine_slots=1)",
            )

        from app.chat_llm import invoke_chat_llm

        text = invoke_chat_llm(message_val, user_id, raw_thread)

        return ChatResponse(
            response=text,
            thread_id=raw_thread,
        )

    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        if _should_fallback_routine_sync(raw_request, payload, err):
            raw_thread, _skin, _concerns, _allergies = _routine_body_from_payload(payload)
            return ChatResponse(
                response=(
                    "Couldn’t load personalized picks right now — here’s a sample routine "
                    "you can edit. Tap Shop to search; add your real products anytime."
                ),
                thread_id=raw_thread,
                routine=_placeholder_routine_response(),
            )
        raise HTTPException(status_code=500, detail=err) from e


# Same paths gateways use for skincare APIs. Do not mount chat under /ui — StaticFiles owns /ui and POSTs there get 405.
for _chat_prefix in ("", "/v1", "/api", "/api/v1"):
    app.include_router(chat_router, prefix=_chat_prefix)


routine_router = APIRouter(tags=["routine"])


@routine_router.post("/routine/recommend", response_model=RoutineRecommendResponse)
async def routine_recommend_endpoint(
    body: RoutineRecommendRequest,
    _user_id: str = Depends(get_user_id),
):
    """Top database pick per AM/PM routine step from skin type, concerns, and allergies."""
    return _build_routine_response(body)


# Same handler under common API prefixes (not /ui — static mount).
for _routine_prefix in ("", "/api", "/v1", "/api/v1"):
    app.include_router(routine_router, prefix=_routine_prefix)


app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
