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
    HUMAN = "human"
    AI = "ai"

class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sender_type = Column(Enum(SenderType), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    chat_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )

    chat = relationship("Chat", back_populates="messages")
    report_md = Column(Text, nullable=True)
