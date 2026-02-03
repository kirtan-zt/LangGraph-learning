from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Sequence
from uuid import UUID
from sqlalchemy import delete
from app.models.document import Document
from app.models.chunk import Chunk

class DocumentRepository:
    """Storage logic for Documents
    """
    async def create(
        self,
        session: AsyncSession,
        document: Document,
    ) -> Document:
        """Data retrieval for documents

        Args:
            session (AsyncSession): Database instance
            document (Document): Dictionary that maps to Document model

        Returns:
            Document: Database model that stores document information.
        """
        session.add(document)
        await session.commit()
        await session.refresh(document)
        return document

    async def find_all(
        self,
        session: AsyncSession,
    ) -> Sequence[Document]:
        """Lists all documents from storage

        Args:
            session (AsyncSession): Database instance

        Returns:
            Sequence[Document]: JSON array of documents objects.
        """
        result = await session.execute(select(Document))
        return result.scalars().all()
    
    async def get_by_id(
        self,
        session: AsyncSession,
        document_id: UUID,
    ) -> Document | None:
        """Retrieve a specific document by it's ID

        Args:
            session (AsyncSession): Database instance
            document_id (UUID): Unique document id

        Returns:
            Document: Database model of document
        """
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_ids(
        self,
        session: AsyncSession,
        ids: Sequence[UUID],
    ) -> list[Document]:
        """Retrieve multile documents by their ID

        Args:
            session (AsyncSession): Database instance.
            ids (Sequence[UUID]): Unique id for searching

        Returns:
            list[Document]: A JSON array of document object.
        """
        if not ids:
            return []

        result = await session.execute(
            select(Document).where(Document.id.in_(list(ids)))
        )
        return result.scalars().all()
    
    async def delete_by_id(
        self,
        session: AsyncSession,
        document_id: UUID,
    ) -> bool:
        """Delete a document by it's ID

        Args:
            session (AsyncSession): Database instance.
            document_id (UUID): Unique document id to be deleted

        Returns:
            bool: Status of successful deletion (True) or failure to delete (False)
        """
        await session.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )

        result = await session.execute(
            delete(Document).where(Document.id == document_id)
        )

        return result.rowcount > 0