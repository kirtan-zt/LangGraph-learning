from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import List, Optional
from enum import Enum

class SenderType(str, Enum):
    human = "human"
    ai = "ai"

class MessageCreate(BaseModel):
    content: str = Field(description="Message content")

class MessageRead(BaseModel):
    id: UUID
    chat_id: UUID
    sender_type: SenderType
    content: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True