from fastapi import APIRouter, Depends, HTTPException, status, Response
from uuid import UUID
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_db
from app.schemas.chat import ChatCreate, ChatRead, ChatResponse
from app.schemas.logs import MessageCreate, MessageRead
from app.services.chat import ChatService
from app.core.dependencies import get_chat_svc, get_message_repo
import logging
from app.repositories.logs import MessageRepository
from app.models.api_response import StandardResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chats", tags=["chats"])

@router.post("/", response_model=StandardResponse[ChatRead])
async def create_chat(
    chat_create: ChatCreate,
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(get_chat_svc),
):
    """
    Initializes a new chat session associated with specified document resources.

    Args:
        chat_create: Configuration for the new chat, including document references.
        db: Asynchronous database session.
        chat_svc: Service handling chat business logic.

    Returns:
        ChatRead: The newly created chat session record.
    """
    try:
        chat_data = await chat_svc.create_chat(db, chat_create)
        return StandardResponse(
            status=201,
            message="New chat created successfully",
            data=chat_data
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

@router.get("/{chat_id}/report")
async def download_report(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    message_repo: MessageRepository = Depends(get_message_repo),
):
    """
    Exports the latest AI-generated summary report for a specific chat.

    Returns a Markdown file via stream if the report exists.

    Args:
        chat_id: Unique identifier of the chat session.
        db: Asynchronous database session.
        message_repo: Repository for message and report retrieval.

    Raises:
        HTTPException: 404 if no report has been generated for the session.
    """
    message = await message_repo.get_latest_ai_message(db, chat_id)

    if not message or not message.report_md:
        raise HTTPException(
            status_code=404,
            detail="No report available for this chat",
        )

    return Response(
        content=message.report_md,
        media_type="text/markdown",
        headers={
            "Content-Disposition": "attachment; filename=report.md"
        },
    )

@router.get("/", response_model=StandardResponse[List[ChatRead]])
async def get_chats(
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(get_chat_svc),
):
    """
    Retrieves a paginated list of all historical chat sessions.
    """
    get_chat_data = await chat_svc.find_all_chats(db)
    return StandardResponse(
        status=200,
        message="Chat sessions retrieved successfully",
        data=get_chat_data
    )


@router.post("/{chat_id}/messages", response_model=StandardResponse[ChatResponse])
async def send_message(
    chat_id: UUID,
    payload: MessageCreate,
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(get_chat_svc),
):
    """
    Processes a user query through the RAG pipeline to generate an AI response.

    This endpoint orchestrates document retrieval, LLM context injection, 
    and response persistence.

    Args:
        chat_id: ID of the active chat session.
        payload: The user's message content.

    Returns:
        ChatResponse: AI-generated text, source citations, and confidence metrics.

    Raises:
        HTTPException: 400 for validation errors.
        HTTPException: 500 for runtime LLM or database failures.
    """
    try:
        message = await chat_svc.send_message(
            session=db,
            chat_id=chat_id,
            message_create=payload,
        )
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate AI response",
            )
        chat_response_data = ChatResponse(
            answer=message.content,
            references=message.sources,
            confidence=message.confidence,
        )
        return StandardResponse(
            status=200,
            message="LLM response generated",
            data=chat_response_data
        )

    except ValueError as e:
        logger.warning(f"Chat validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except HTTPException:
        raise

    except Exception as e:
        # Unexpected errors (LLM, DB, vector store, etc.)
        logger.exception("Unexpected error while generating chat response")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request",
        )


@router.get("/{chat_id}/messages", response_model=StandardResponse[List[MessageRead]])
async def get_messages(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(get_chat_svc),
):
    """
    Retrieves the chronological message history for a specific chat session.
    """
    get_message_data = await chat_svc.find_messages(db, chat_id)
    return StandardResponse(
        status=200,
        message=f"Messages retrieved for chat ID {chat_id} successfully",
        data=get_message_data
    )
