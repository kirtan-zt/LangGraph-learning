from sqlalchemy import Column, DateTime, LargeBinary, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from app.core.base import Base
from sqlalchemy.sql import func

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String, nullable=False)
    content = Column(LargeBinary, nullable=False)
    upload_date = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chats = relationship(
        "Chat",
        secondary="chat_file_links",
        back_populates="files",
    )
