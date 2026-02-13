from typing import Annotated
from fastapi import Depends
from app.core.app_context import create_app_context
from app.services import DocumentService, AIService, ChatService
from app.repositories.logs import MessageRepository
from app.core.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession

# Initialize the global application context for service singleton management
ctx = create_app_context()

# Database Dependencies
# Injected database session with automatic cleanup via get_db generator
SessionDep = Annotated[AsyncSession, Depends(get_db)]

# Service Provider Functions 
def get_document_svc() -> DocumentService:
    """
    Returns the singleton DocumentService instance from the app context.
    """
    return ctx.document_svc

def get_ai_svc() -> AIService:
    """
    Returns the singleton AIService instance for LLM and embedding operations.
    """
    return ctx.ai_svc

def get_chat_svc() -> ChatService:
    """
    Returns the singleton ChatService instance for conversation orchestration.
    """
    return ctx.chat_svc

def get_message_repo() -> MessageRepository:
    """
    Factory function for MessageRepository to handle persistence logic.
    """
    return MessageRepository()

# Annotated Dependency Aliases : To simplify route signatures and improve readability in endpoint definitions.
DocumentSvcDep = Annotated[DocumentService, Depends(get_document_svc)]
AISvcDep = Annotated[AIService, Depends(get_ai_svc)]
ChatSvcDep = Annotated[ChatService, Depends(get_chat_svc)]

# Define public interface for the dependencies module
__all__ = [
    "SessionDep",
    "DocumentSvcDep",
    "AISvcDep",
    "ChatSvcDep",
]
