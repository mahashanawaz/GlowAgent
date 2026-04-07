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

AUTH0_DOMAIN   = os.environ.get("AUTH0_DOMAIN")   
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE")  

security = HTTPBearer()


@lru_cache(maxsize=1)
def get_jwks() -> dict:
    """
    Fetch Auth0's public JWKS (JSON Web Key Set).
    Auth0 signs tokens with its private key; we verify with the public key.
    Cached so we only fetch once per process startup.
    """
    url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def verify_token(token: str) -> dict:
    """
    Decode and verify an Auth0 JWT.
    Returns the decoded payload (claims) if valid.
    Raises HTTPException 401 if anything is wrong.
    """
    try:
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
            audience=AUTH0_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/",
        )
        return payload

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