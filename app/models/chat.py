from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from app.core.base import Base

class Chat(Base):
    """
    Represents a unique chat session or conversation thread.
    
    This model serves as the parent entity for messages and maintains 
    a many-to-many relationship with uploaded documents.
    """
    __tablename__ = "chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String, nullable=False)

    # Many-to-many relationship with Document through the junction table
    files = relationship(
        "Document",
        secondary="chat_file_links",
        back_populates="chats",
    )

    # One-to-many relationship with Message: 'delete-orphan' ensures messages are purged when a Chat is deleted
    messages = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
    )


class ChatFileLink(Base):
    """
    Junction table representing the many-to-many association 
    between Chats and Documents.
    
    Implements cascading deletes to ensure referential integrity when 
    either a chat or a document is removed.
    """
    __tablename__ = "chat_file_links"

    chat_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        primary_key=True,
    )
    file_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
