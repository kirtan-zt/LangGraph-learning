from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from typing import Sequence
from sqlalchemy.orm import selectinload
from app.models.chat import Chat
from app.models.document import Document

class ChatRepository:
    """Storage logic for chats
    """
    async def create(
        self,
        db: AsyncSession,
        chat: Chat,
        documents: Sequence[Document],
    ) -> Chat:
        """Data retrieval for chats

        Args:
            db (AsyncSession): Database instance
            chat (Chat): Dictionary that maps to Chat model
            documents (Sequence[Document]): Document reference object for mapping chat id with document id

        Returns:
            Chat: Database model that stores chat contents.
        """
        chat.files = list(documents)
        db.add(chat)
        await db.flush()
        await db.commit() 
        await db.refresh(chat)
        return chat

    async def find_all(self, db: AsyncSession) -> Sequence[Chat]:
        """Lists all chats from storage

        Args:
            db (AsyncSession): Database instance.

        Returns:
            Sequence[Chat]: JSON array of chat objects.
        """
        result = await db.execute(select(Chat))
        return result.scalars().all()

    async def get_by_id(self, db: AsyncSession, chat_id: UUID) -> Chat | None:
        """Retrieve a specific chat by it's ID, using selectinload to prevent Lazy Loading

        Args:
            db (AsyncSession): Database instance.
            chat_id (UUID): Unique chat id

        Returns:
            Chat: Database model of chat
        """
        result = await db.execute(
            select(Chat)
            .options(selectinload(Chat.files))  
            .where(Chat.id == chat_id)
        )
        return result.scalar_one_or_none()
