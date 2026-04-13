"""PerfectCorp skin analysis integration for visible concern inference."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

import httpx
from fastapi import HTTPException, status

FILE_API_URL = "https://yce-api-01.makeupar.com/s2s/v2.0/file/skin-analysis"
TASK_API_URL = "https://yce-api-01.makeupar.com/s2s/v2.0/task/skin-analysis"
POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 45.0

PROFILE_CONCERN_TO_ACTIONS = {
    "acne": ["acne"],
    "pores": ["pore"],
    "dryness": ["moisture"],
    "sensitivity": [],
    "dark spots": ["age_spot"],
    "aging": ["wrinkle"],
    "oil": ["oiliness"],
    "redness": ["redness"],
    "hyperpigmentation": ["age_spot"],
    "texture": ["texture"],
}

CONCERN_LABELS = {
    "wrinkle": "fine lines and wrinkles",
    "pore": "visible pores",
    "texture": "uneven texture",
    "acne": "blemishes or acne-like spots",
    "moisture": "dryness or dehydration",
    "age_spot": "dark spots or pigmentation",
    "oiliness": "excess oiliness",
    "redness": "visible redness",
}

ALLOWED_RESULT_TYPES = set(CONCERN_LABELS.keys())


def get_api_key() -> str:
    api_key = (os.getenv("PERFECTCORP_API_KEY")).strip()
    return api_key


def _configured_dst_actions() -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for actions in PROFILE_CONCERN_TO_ACTIONS.values():
        for action in actions:
            if action not in seen:
                seen.add(action)
                deduped.append(action)
    if "wrinkle" not in seen:
        deduped.append("wrinkle")
    return deduped


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _task_error_detail(payload: dict[str, Any], data: dict[str, Any]) -> str:
    pieces: list[str] = []
    for source in (data, payload):
        if not isinstance(source, dict):
            continue
        for key in ("error", "detail", "message", "error_code", "reason"):
            value = source.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in pieces:
                pieces.append(text)
    if pieces:
        return " | ".join(pieces)
    return "Skin analysis service returned an error while processing the image."


def decode_image_payload(image_payload: dict[str, Any]) -> tuple[bytes, str, str]:
    data_url = str(image_payload.get("data_url") or image_payload.get("dataUrl") or "").strip()
    content_type = str(
        image_payload.get("content_type") or image_payload.get("contentType") or "image/jpeg"
    ).strip() or "image/jpeg"
    file_name = str(
        image_payload.get("file_name") or image_payload.get("fileName") or "skin-analysis.jpg"
    ).strip() or "skin-analysis.jpg"

    if not data_url:
        raise HTTPException(status_code=400, detail="Image data is required for skin analysis.")

    if data_url.startswith("data:"):
        try:
            meta, encoded = data_url.split(",", 1)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid image data URL.") from exc
        if ";base64" not in meta:
            raise HTTPException(status_code=400, detail="Skin analysis image must be base64 encoded.")
        meta_content_type = meta[5:].split(";")[0].strip()
        if meta_content_type:
            content_type = meta_content_type
    else:
        encoded = data_url

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.") from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    return image_bytes, content_type, file_name


async def _create_file_upload(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    file_name: str,
    content_type: str,
    file_size: int,
) -> tuple[str, dict[str, str], str]:
    res = await client.post(
        FILE_API_URL,
        headers=_auth_headers(api_key),
        json={
            "files": [
                {
                    "content_type": content_type,
                    "file_name": file_name,
                    "file_size": file_size,
                }
            ]
        },
    )
    res.raise_for_status()
    payload = res.json()
    try:
        file_info = payload["data"]["files"][0]
        request_info = file_info["requests"][0]
        return file_info["file_id"], dict(request_info.get("headers") or {}), request_info["url"]
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Skin analysis upload metadata response was missing file upload details.",
        ) from exc


async def _upload_file_bytes(
    client: httpx.AsyncClient,
    *,
    upload_url: str,
    upload_headers: dict[str, str],
    image_bytes: bytes,
) -> None:
    res = await client.put(upload_url, headers=upload_headers, content=image_bytes)
    res.raise_for_status()


async def _create_skin_task(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    file_id: str,
    dst_actions: list[str],
) -> str:
    res = await client.post(
        TASK_API_URL,
        headers=_auth_headers(api_key),
        json={
            "src_file_id": file_id,
            "dst_actions": dst_actions,
            "format": "json",
        },
    )
    res.raise_for_status()
    payload = res.json()
    try:
        return str(payload["data"]["task_id"])
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Skin analysis task creation response was missing a task_id.",
        ) from exc


async def _poll_skin_task(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    task_id: str,
) -> dict[str, Any]:
    elapsed = 0.0
    while elapsed <= POLL_TIMEOUT_SECONDS:
        res = await client.get(
            f"{TASK_API_URL}/{task_id}",
            headers=_auth_headers(api_key),
        )
        res.raise_for_status()
        payload = res.json()
        data = payload.get("data") or {}
        task_status = str(data.get("task_status") or "").lower()
        if task_status == "success":
            return data
        if task_status == "error":
            raise HTTPException(
                status_code=502,
                detail=_task_error_detail(payload, data),
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
    raise HTTPException(
        status_code=504,
        detail="Skin analysis timed out before results were ready.",
    )


async def analyze_skin_image(image_payload: dict[str, Any]) -> list[dict[str, Any]]:
    api_key = get_api_key()
    image_bytes, content_type, file_name = decode_image_payload(image_payload)
    dst_actions = _configured_dst_actions()

    timeout = httpx.Timeout(30.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            file_id, upload_headers, upload_url = await _create_file_upload(
                client,
                api_key,
                file_name=file_name,
                content_type=content_type,
                file_size=len(image_bytes),
            )
            await _upload_file_bytes(
                client,
                upload_url=upload_url,
                upload_headers=upload_headers,
                image_bytes=image_bytes,
            )
            task_id = await _create_skin_task(
                client,
                api_key,
                file_id=file_id,
                dst_actions=dst_actions,
            )
            result_data = await _poll_skin_task(client, api_key, task_id=task_id)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or str(exc)
            raise HTTPException(
                status_code=502,
                detail=f"Skin analysis provider error: {detail}",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Skin analysis provider request failed: {str(exc)}",
            ) from exc

    output = ((result_data.get("results") or {}).get("output") or [])
    cleaned: list[dict[str, Any]] = []
    for item in output:
        concern_type = str(item.get("type") or "").strip()
        if not concern_type or concern_type not in ALLOWED_RESULT_TYPES:
            continue
        cleaned.append(
            {
                "type": concern_type,
                "label": CONCERN_LABELS.get(concern_type, concern_type.replace("_", " ")),
                "ui_score": item.get("ui_score"),
                "raw_score": item.get("raw_score"),
                "concern_signal": max(0.0, 100.0 - float(item.get("ui_score") or 0)),
                "mask_urls": item.get("mask_urls") or [],
            }
        )
    cleaned.sort(key=lambda row: float(row.get("concern_signal") or 0), reverse=True)
    return cleaned


def _visible_concern_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = [row for row in results if float(row.get("concern_signal") or 0) >= 20]
    if visible:
        return visible[:4]
    return results[:2]


def summarize_skin_analysis(results: list[dict[str, Any]]) -> str:
    if not results:
        return (
            "I reviewed the uploaded photo, but the skin analysis service did not return "
            "any visible concern results. This is not a diagnosis."
        )

    visible = _visible_concern_rows(results)

    lines = [
        "I analyzed the uploaded photo for visible skin concerns.",
        "This is not a diagnosis. It is only an image-based summary of visible patterns.",
        "",
        "Most visible concerns:",
    ]
    for row in visible[:4]:
        score = int(round(float(row.get("concern_signal") or 0)))
        label = row.get("label") or row.get("type") or "visible concern"
        lines.append(f"- {label}: {score}/100 visible signal")

    lines.extend(
        [
            "",
            "I can use these visible concerns to guide product suggestions and routine steps.",
        ]
    )
    return "\n".join(lines)


def concerns_for_agent(results: list[dict[str, Any]]) -> list[str]:
    concerns: list[str] = []
    for row in _visible_concern_rows(results):
        label = str(row.get("label") or row.get("type") or "").strip()
        if label and label not in concerns:
            concerns.append(label)
    return concerns
