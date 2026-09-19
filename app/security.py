"""Security utilities and API key authentication."""

import os
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from app.config import get_settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key from request headers."""
    settings = get_settings()
    expected_key = os.getenv("API_KEY") or settings.api_key

    if not expected_key:
        return "dev-key"

    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
