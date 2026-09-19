"""Pydantic schemas and data models."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or message")
    user_id: Optional[str] = Field(default=None, description="Optional user ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    cached: bool = Field(default=False, description="Whether response came from cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str
