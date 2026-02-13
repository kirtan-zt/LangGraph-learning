from sqlalchemy import Column, Text, DateTime, Enum, JSON
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from datetime import datetime, timezone
from enum import Enum as PyEnum
from sqlalchemy import Float
from app.core.base import Base
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class SenderType(PyEnum):
    """
    Categorizes the originator of a message within a chat session.
    """
    HUMAN = "human"
    AI = "ai"

class Message(Base):
    """
    Represents an individual exchange in a chat session.
    
    Stores the textual content, AI-generated metadata (sources and confidence),
    and links the message to a specific Chat session.
    """
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Discriminator for distinguishing between user input and model output
    sender_type = Column(Enum(SenderType), nullable=False)

    # The primary text content of the message
    content = Column(Text, nullable=False)

    # JSON-encoded metadata containing references or citations used by the AI
    sources = Column(JSON, nullable=True)

    # Probability or similarity score associated with the AI response
    confidence = Column(Float, nullable=True)

    # Client-side timestamp for message creation in UTC
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Foreign key link to the parent Chat session
    # CASCADE ensures messages are deleted alongside their parent chat
    chat_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Relationship back to the Chat parent entity
    chat = relationship("Chat", back_populates="messages")

    # Optional field for storing a formatted Markdown report or summary
    report_md = Column(Text, nullable=True)
