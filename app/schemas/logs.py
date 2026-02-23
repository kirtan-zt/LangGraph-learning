from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import List, Optional
from enum import Enum

class SenderType(str, Enum):
    """
    Specifies the origin of a message to facilitate conditional UI rendering.
    """
    human = "human"
    ai = "ai"

class MessageCreate(BaseModel):
    """
    Data transfer object for capturing new user input.
    """
    content: str = Field(description="Message content")

class MessageRead(BaseModel):
    """
    Comprehensive view of a message, including RAG-specific metadata.
    """
    id: UUID
    chat_id: UUID
    sender_type: SenderType
    content: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)