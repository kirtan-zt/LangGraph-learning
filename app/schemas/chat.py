from pydantic import BaseModel, Field
from uuid import UUID
from typing import List
from app.schemas.document import DocumentRead

class ChatCreate(BaseModel):
    document_ids: List[UUID] = Field(
        default_factory=list,
        description="Documents associated with this chat",
    )

class ChatRead(BaseModel):
    id: UUID
    name: str
    documents: List[DocumentRead] = Field(default_factory=list)

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    answer: str
    references: List[str]
    confidence: float