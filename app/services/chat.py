from dataclasses import dataclass
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.chat import ChatCreate
from app.models.chat import Chat
from app.models.logs import Message
from app.repositories.chat import ChatRepository
from app.repositories.document import DocumentRepository
from app.repositories.logs import MessageRepository
from app.services.ai import AIService
from app.services.document import DocumentService
from app.schemas.logs import MessageCreate
from app.models.logs import SenderType
from app.langgraph.state import ResearchState
from app.langgraph.task_classifier import classify_task
from langchain_core.documents import Document as LCDocument
from langsmith import traceable

@dataclass
class ChatService:
    """
    Coordinates the lifecycle of chat sessions, document retrieval orchestration, 
    and the execution of the RAG research pipeline.
    """       
    chat_repository: ChatRepository
    message_repository: MessageRepository
    ai_svc: AIService
    document_svc: DocumentService
    research_graph: any
    
    async def create_chat(
    self,
    session: AsyncSession,
    chat_create: ChatCreate,
) -> Chat:
        """
        Initializes a new chat session linked to a set of validated documents.

        Args:
            session: Asynchronous database session.
            chat_create: Request schema containing document associations.

        Returns:
            Chat: The persisted Chat model instance.

        Raises:
            ValueError: If document IDs are missing or non-existent.
        """
        if not chat_create.document_ids:
            raise ValueError("At least one document_id is required")

        documents = await self.document_svc.get_by_ids(
            session=session,
            document_ids=chat_create.document_ids,
        )

        if len(documents) != len(chat_create.document_ids):
            raise ValueError("One or more document_ids are invalid")

        chat = Chat(name="New Chat")
        return await self.chat_repository.create(
            db=session,
            chat=chat,
            documents=documents,
        )

    async def find_all_chats(
        self,
        session: AsyncSession,
    ) -> list[Chat]:
        """
        Retrieves all historical chat sessions from the repository.
        """
        return await self.chat_repository.find_all(session)
        
    async def retrieve_balanced_documents(
        self,
        query: str,
        document_ids: list[UUID],
        k_per_doc: int = 3,
    ) -> list[LCDocument]:
        """
        Performs per-document similarity search to ensure context is sampled 
        evenly across all associated files.

        Args:
            query: The search query string.
            document_ids: List of specific document IDs to query.
            k_per_doc: Number of chunks to retrieve per individual document.
        """

        all_docs: list[LCDocument] = []

        for doc_id in document_ids:
            docs = await self.document_svc.search(
                query=query,
                file_ids=[doc_id],   
                k=k_per_doc,
            )
            all_docs.extend(docs)

        return all_docs
    @traceable(name="chat_request")
    async def send_message(
        self,
        session: AsyncSession,
        chat_id: UUID,
        message_create: MessageCreate,
    ) -> Message:
        """
        Orchestrates the full RAG pipeline: task classification, retrieval, 
        LangGraph processing, and persistence.

        Args:
            chat_id: The ID of the session receiving the message.
            message_create: Schema containing user input.

        Returns:
            Message: The finalized AI response record.
        """
        chat = await self.chat_repository.get_by_id(session, chat_id)
        if chat is None:
            raise ValueError(f"Chat {chat_id} not found")
        
        # Determine if the query requires standard retrieval or specialized analysis
        task_type = await classify_task(self.ai_svc.llm, message_create.content)
        
        # Persist user input immediately
        user_msg = Message(
            chat_id=chat_id,
            content=message_create.content,
            sender_type=SenderType.HUMAN,
        )
        session.add(user_msg)

        document_ids = [doc.id for doc in chat.files]

        # Strategic context retrieval based on task complexity
        if task_type in {"summarize", "compare", "insights"}:
            context_docs = await self.retrieve_balanced_documents(
                query=message_create.content,
                document_ids=document_ids,
                k_per_doc=3,
            )
        else:
            context_docs = await self.document_svc.search(
                query=message_create.content,
                file_ids=document_ids,
                k=5,
            )
        if not context_docs:
            return Message(
                chat_id=chat_id,
                sender_type=SenderType.AI,
                content="The provided documents do not contain information to answer this question.",
                sources=[],
                confidence=0.15,
            )

        # Generate RAG result
        rag_result = await self.ai_svc.generate_rag_answer(
            question=message_create.content,
            documents=context_docs
            )

        if rag_result.confidence < 0.2:
            rag_result.sources = []

        # Trigger the state-graph workflow for multi-step reasoning
        state = ResearchState(
            question=message_create.content,
            documents=context_docs,
            task_type=task_type,
        )
        final_state = await self.research_graph.ainvoke(state)

        # Map graph output back to the database Message model
        ai_msg = Message(
            chat_id=chat_id,
            sender_type=SenderType.AI,
            content = final_state.get("answer") or final_state.get("report_md"),
            sources = final_state.get("sources", []),
            confidence = final_state.get("confidence", 0.0),
            report_md=final_state.get("report_md"),
        )
        session.add(ai_msg)

        # Update chat title if it is the first exchange
        if chat.name == "New Chat":
            chat.name = await self.ai_svc.generate_chat_title(
                question=message_create.content,
                answer=ai_msg.content,
            )
        
        await session.commit()
        await session.refresh(ai_msg)
        return ai_msg
    
    async def find_messages(
        self,
        session: AsyncSession,
        chat_id: UUID,
    ) -> list[Message]:
        """
        Retrieves the full message history for a given chat ID.
        """
        return await self.message_repository.find_by_chat_id(session, chat_id)