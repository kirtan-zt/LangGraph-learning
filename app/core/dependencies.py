from typing import Annotated
from fastapi import Depends
from app.core.app_context import create_app_context
from app.services import DocumentService, AIService, ChatService
from app.repositories.logs import MessageRepository
from app.core.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession

ctx = create_app_context()

# DB 
SessionDep = Annotated[AsyncSession, Depends(get_db)]

# Services
def get_document_svc() -> DocumentService:
    return ctx.document_svc

def get_ai_svc() -> AIService:
    return ctx.ai_svc

def get_chat_svc() -> ChatService:
    return ctx.chat_svc

def get_message_repo() -> MessageRepository:
    return MessageRepository()

DocumentSvcDep = Annotated[DocumentService, Depends(get_document_svc)]
AISvcDep = Annotated[AIService, Depends(get_ai_svc)]
ChatSvcDep = Annotated[ChatService, Depends(get_chat_svc)]

__all__ = [
    "SessionDep",
    "DocumentSvcDep",
    "AISvcDep",
    "ChatSvcDep",
]
