from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from typing import List
from app.schemas.document import DocumentRead

class ChatCreate(BaseModel):
    """
    Data transfer object for initializing a chat session.
    """
    document_ids: List[UUID] = Field(
        default_factory=list,
        description="Documents associated with this chat",
    )

class ChatRead(BaseModel):
    """
    Schema for representing a chat session's metadata and associated resources.
    """
    id: UUID
    name: str
    files: List[DocumentRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

class ChatResponse(BaseModel):
    """
    Schema for the final AI-generated response in a RAG pipeline.
    
    Includes the synthesized answer along with traceability metadata.
    """
    answer: str
    references: List[str]
    confidence: float