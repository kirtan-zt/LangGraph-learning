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

@dataclass
class ChatService:
    """Business logic for chat model"""       
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
        """Method to create chat id

        Args:
            session (AsyncSession): Database instance
            chat_create (ChatCreate): Request model to store chat id

        Returns:
            Chat: Dictionary that maps to Chat database 
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
        """Lists all chat ids from the database.

        Args:
            session (AsyncSession): Database instance.

        Returns:
            list[Chat]: A JSON array of chat objects
        """
        return await self.chat_repository.find_all(session)
        
    async def retrieve_balanced_documents(
        self,
        query: str,
        document_ids: list[UUID],
        k_per_doc: int = 3,
    ) -> list[LCDocument]:
        """Per-document chunk retrieval logic 

        Args:
            query (str): User query
            document_ids (list[UUID]): List of reference documents
            k_per_doc (int, optional): Top-k results, defaults to 3

        Returns:
            list[LCDocument]: _description_
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

    async def send_message(
        self,
        session: AsyncSession,
        chat_id: UUID,
        message_create: MessageCreate,
    ) -> Message:
        """Method to send prompt to LLM from the user.

        Args:
            session (AsyncSession): Database instance.
            chat_id (UUID): Unique chat id
            message_create (MessageCreate): Request model object

        Raises:
            ValueError: Chat id validation check

        Returns:
            Message: LLM response for the given question
        """
        chat = await self.chat_repository.get_by_id(session, chat_id)
        if chat is None:
            raise ValueError(f"Chat {chat_id} not found")
        
        task_type = await classify_task(self.ai_svc.llm, message_create.content)
        # Save user message
        user_msg = Message(
            chat_id=chat_id,
            content=message_create.content,
            sender_type=SenderType.HUMAN,
        )
        session.add(user_msg)

        history = await self.get_chat_history(session, chat_id)
        document_ids = [doc.id for doc in chat.files]

        # Retrieve context from documents
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
            raise ValueError("No relevant content found in the selected documents")

        # LangGraph invocation 
        state = ResearchState(
            question=message_create.content,
            documents=context_docs,
            task_type=task_type,
        )
        final_state = await self.research_graph.ainvoke(state)

        # Save AI message
        ai_msg = Message(
            chat_id=chat_id,
            sender_type=SenderType.AI,
            content = final_state.get("answer") or final_state.get("report_md"),
            sources = final_state.get("sources", []),
            confidence = final_state.get("confidence", 0.0),
            report_md=final_state.get("report_md"),
        )
        session.add(ai_msg)
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
        """Lists all messages in the conversation.

        Args:
            session (AsyncSession): Database instance.
            chat_id (UUID): Unique chat id

        Returns:
            list[Message]: A JSON array of message objects.
        """
        return await self.message_repository.find_by_chat_id(session, chat_id)
    
    async def get_chat_history(
    self,
    session: AsyncSession,
    chat_id: UUID,
    limit: int = 6,
    ) -> str:
        messages = await self.message_repository.find_by_chat_id(session, chat_id)

        # last N messages
        recent = messages[-limit:]

        history = []
        for msg in recent:
            role = "User" if msg.sender_type == SenderType.HUMAN else "Assistant"
            history.append(f"{role}: {msg.content}")

        return "\n".join(history)
