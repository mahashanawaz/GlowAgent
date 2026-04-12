"""
app/auth.py
-----------
Auth0 JWT verification for GlowAgent's FastAPI backend.

Every protected endpoint depends on get_current_user(),
which extracts the user's Auth0 sub (unique ID) from the JWT.
"""

import os
import httpx
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

AUTH0_DOMAIN   = os.environ.get("AUTH0_DOMAIN")   
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE")  
GUEST_TOKEN_PREFIX = "glow-guest:"

security = HTTPBearer()


def _guest_user_from_token(token: str) -> Optional[dict]:
    """
    Accept a lightweight guest bearer token created by the SPA for session-only usage.
    This lets guest mode use the same API flow as signed-in mode without persisting data.
    """
    if not token or not token.startswith(GUEST_TOKEN_PREFIX):
        return None
    guest_id = token[len(GUEST_TOKEN_PREFIX) :].strip()
    if not guest_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid guest token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "sub": f"guest|{guest_id}",
        "name": "Guest",
        "guest": True,
    }


def _require_auth0_settings() -> tuple[str, str]:
    """
    Ensure required Auth0 settings exist and normalize the domain format.
    """
    if not AUTH0_DOMAIN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth configuration error: AUTH0_DOMAIN is not set.",
        )
    if not AUTH0_AUDIENCE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth configuration error: AUTH0_AUDIENCE is not set.",
        )

    domain = AUTH0_DOMAIN.replace("https://", "").replace("http://", "").strip("/")
    return domain, AUTH0_AUDIENCE


@lru_cache(maxsize=1)
def get_jwks() -> dict:
    """
    Fetch Auth0's public JWKS (JSON Web Key Set).
    Auth0 signs tokens with its private key; we verify with the public key.
    Cached so we only fetch once per process startup.
    """
    domain, _ = _require_auth0_settings()
    url = f"https://{domain}/.well-known/jwks.json"
    try:
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to reach Auth0 JWKS endpoint at {url}: {str(e)}",
        ) from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Auth0 JWKS endpoint returned {e.response.status_code}.",
        ) from e
    return response.json()


def verify_token(token: str) -> dict:
    """
    Decode and verify an Auth0 JWT.
    Returns the decoded payload (claims) if valid.
    Raises HTTPException 401 if anything is wrong.
    """
    guest_user = _guest_user_from_token(token)
    if guest_user is not None:
        return guest_user
    try:
        domain, audience = _require_auth0_settings()
        # Get the signing key that matches this token's 'kid' header
        jwks = get_jwks()
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}

        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n":   key["n"],
                    "e":   key["e"],
                }
                break

        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find matching signing key",
            )

        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=f"https://{domain}/",
        )
        return payload

    except HTTPException:
        raise
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    FastAPI dependency — inject this into any route that requires auth.

    Usage:
        @app.post("/chat")
        async def chat(request: ChatRequest, user=Depends(get_current_user)):
            user_id = user["sub"]   # Auth0's unique user ID (e.g. "auth0|abc123")
            ...
    """
    return verify_token(credentials.credentials)


def get_user_id(user: dict = Depends(get_current_user)) -> str:
    """Shortcut dependency that returns just the user's Auth0 sub string."""
    return user["sub"]
