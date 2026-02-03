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

@dataclass
class ChatService:
    """Business logic for chat model"""
    chat_repository: ChatRepository
    message_repository: MessageRepository
    ai_svc: AIService
    document_svc: DocumentService

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
        context_docs = await self.document_svc.search(
            query=message_create.content,
            file_ids=document_ids,
            k=5,
        )
        if not context_docs:
            raise ValueError("No relevant content found in the selected documents")

        # Generate AI response
        rag_result = await self.ai_svc.generate_rag_answer(
            question=message_create.content,
            documents=context_docs,
            history=history,
        )

        # Save AI message
        ai_msg = Message(
            chat_id=chat_id,
            sender_type=SenderType.AI,
            content=rag_result.answer,
            sources=rag_result.sources,
            confidence=rag_result.confidence,
        )
        session.add(ai_msg)
        if chat.name == "New Chat":
            title = await self.ai_svc.generate_chat_title(
                question=message_create.content,
                answer=rag_result.answer,
            )
            chat.name = title
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