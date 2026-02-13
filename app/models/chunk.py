from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.base import Base

class Chunk(Base):
    """
    Represents a granular segment of text extracted from a Document.
    
    This model stores individual text fragments used for semantic indexing 
    and context retrieval. It maintains the original document order via 
    the `chunk_index`.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)

    # Reference to the source document; cascade delete ensures orphans are not left when a document is removed from the system.
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # The sequential position of the chunk within the source document (0-indexed)
    chunk_index = Column(Integer, nullable=False)
    
    # The raw text content used for embedding and LLM context injection
    content = Column(Text, nullable=False)

    # Relationship back to the parent Document entity
    document = relationship("Document", back_populates="chunks")