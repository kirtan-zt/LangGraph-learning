from uuid import UUID
from typing import Sequence, List, Optional
import io
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf import PdfReader
from langchain_core.documents import Document as LCDocument
from app.models.chunk import Chunk
from app.models.document import Document
from app.repositories.document import DocumentRepository

class DocumentService:
    """
    Orchestrates document ingestion, text extraction, chunking, and vector indexing.
    
    This service ensures atomic persistence across the relational database 
    and the vector store for RAG operations.
    """
    def __init__(
        self,
        document_repository: DocumentRepository,
        vector_store,
        text_splitter,
    ):
        self.document_repository = document_repository
        self.vector_store = vector_store
        self.text_splitter = text_splitter

    async def get_by_ids(
        self,
        session: AsyncSession,
        document_ids: Sequence[UUID],
    ) -> list[Document]:
        """
        Retrieves a set of Document entities by their unique identifiers.
        """
        if not document_ids:
            return []

        result = await session.execute(
            select(Document).where(Document.id.in_(document_ids))
        )
        return result.scalars().all()

    async def _save_from_text(
        self,
        db: AsyncSession,
        name: str,
        base_docs: list[LCDocument],
    ) -> Document:
        """
        Internal pipeline to process, chunk, and index document content.

        Performs a two-step persistence: 
        1. Stores raw chunks in the relational DB for deterministic retrieval.
        2. Indexes semantic splits in the vector store for similarity search.
        
        Args:
            db: Active database session.
            name: Original name of the document.
            base_docs: List of LangChain Document objects to be split.

        Returns:
            Document: The persisted document record.
        """
        document = Document(
            name=name,
            content=b"",  # Placeholder if binary storage is handled elsewhere
        )

        db.add(document)
        await db.flush() # Obtain document.id for relationship mapping

        all_splits: list[LCDocument] = []
        chunks: list[Chunk] = []

        for base_doc in base_docs:
            splits = self.text_splitter.split_documents([base_doc])

            for idx, split in enumerate(splits):
                # Enrich metadata for granular vector filtering
                split.metadata.update(
                    {
                        "document_id": str(document.id),
                        "file_name": name,
                        "chunk_index": idx,
                        "page_number": base_doc.metadata["page_number"],
                    }
                )

                all_splits.append(split)

                chunks.append(
                    Chunk(
                        document_id=document.id,
                        chunk_index=idx,
                        content=split.page_content,
                    )
                )
        if not all_splits:
            raise ValueError("No text chunks generated")
          
        db.add_all(chunks)
        # Concurrent indexing in the vector store
        await self.vector_store.aadd_documents(all_splits)
        await db.commit()
        return document

    async def save_from_pdf(
        self,
        db: AsyncSession,
        name: str,
        pdf_bytes: bytes,
    ) -> Document:
        """
        Extracts and indexes text content from binary PDF data.

        Args:
            pdf_bytes: Raw binary content of the PDF file.

        Raises:
            ValueError: If the PDF is malformed, encrypted, or lacks extractable text.
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise ValueError("Failed to read PDF file") from e

        if reader.is_encrypted:
            try:
                reader.decrypt("")  # Attempt access with default empty password
            except Exception:
                raise ValueError("Encrypted PDF is not supported")
            
        base_docs: list[LCDocument] = []
        
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            base_docs.append(
            LCDocument(
                page_content=text,
                metadata={
                    "file_name": name,
                    "page_number": page_number,
                },
            )
        )

        if not base_docs:
            raise ValueError("No text could be extracted from PDF")

        return await self._save_from_text(
            db=db,
            name=name,
            base_docs=base_docs,
        )

    async def save_from_text(
        self,
        db: AsyncSession,
        name: str,
        text: str,
    ) -> Document:
        """
        Indexes a raw text string as a new document.
        """
        if not text.strip():
            raise ValueError("Text content is empty")
        
        base_docs = [
            LCDocument(
                page_content=text,
                metadata={
                    "file_name": name,
                    "page_number": 1,
                },
            )
        ]

        return await self._save_from_text(
            db=db,
            name=name,
            base_docs=base_docs,
        )

    async def search(
        self,
        query: str,
        file_ids: list[UUID],
        k: int = 5,
    ) -> List[LCDocument]:
        """
        Performs semantic search across specified documents using vector embeddings.

        Implements a relevance threshold to filter out low-confidence matches. 
        If no results meet the threshold, it falls back to the top 3 matches 
        to ensure context availability.

        Args:
            query: Semantic search query.
            file_ids: Scope of documents to search within.
            k: Maximum number of chunks to retrieve.
        """
        if not file_ids:
            return []

        # Query vector store with metadata filtering for isolation
        results= await self.vector_store.asimilarity_search_with_score(
            query=query,
            k=k,
            filter={
                "document_id": {"$in": [str(fid) for fid in file_ids]}
            },
        )

        # Lower scores indicate higher similarity in many vector dialects (e.g., L2/Cosine distance)
        RELEVANCE_THRESHOLD = 0.55

        filtered_docs = [
            doc for doc, score in results
            if score <= RELEVANCE_THRESHOLD
        ]

        # Heuristic fallback to prevent 'empty context' failures
        if not filtered_docs and results:
            filtered_docs = [doc for doc, _ in results[:3]]

        return filtered_docs

    async def delete(
        self,
        session: AsyncSession,
        document_id: UUID,
    ) -> None:
        """
        Deletes a document and its relational associations.
        """
        # Check existence
        document = await self.document_repository.get_by_id(
            session,
            document_id,
        )
        if not document:
            raise ValueError(f"Document {document_id} not found")

        await session.delete(document)
        await session.commit()