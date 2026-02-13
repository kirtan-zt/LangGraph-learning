from sqlalchemy import Column, DateTime, LargeBinary, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from app.core.base import Base
from sqlalchemy.sql import func

class Document(Base):
    """
    Represents a source file uploaded for processing and chat context.
    
    This model stores the raw binary data of the file and maintains 
    relationships for both the retrieval-augmented generation (RAG) 
    pipeline and the associated chat sessions.
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Original filename including extension
    name = Column(String, nullable=False)

    # Raw file data stored as bytes; intended for PDF, TXT, or similar formats
    content = Column(LargeBinary, nullable=False)

    # Audit timestamp for upload; defaults to server-side current time
    upload_date = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # One-to-many relationship with text segments
    # Deleting a document automatically purges all associated chunks
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    # Many-to-many relationship with Chat sessions
    # Links are managed via the 'chat_file_links' association table
    chats = relationship(
        "Chat",
        secondary="chat_file_links",
        back_populates="files",
    )
