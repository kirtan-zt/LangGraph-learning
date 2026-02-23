from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from app.services.document import DocumentService
from app.services.ai import AIService
from app.services.chat import ChatService
from app.repositories import DocumentRepository, ChatRepository, MessageRepository
from app.langgraph.workflow import build_research_graph

class AppContext:
    """
    App context is a single source of initiating embedding models, LLM models, storing repository for models, etc.
    """
    def __init__(self, settings):
        # Text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        # Embeddings 
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )

        # Vector store
        self.vector_store = Chroma(
            collection_name=settings.vector_store_collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_DB_PATH,
            collection_metadata={
                "space": "cosine", # cosine similarity 
                "M": 64, # Number of connections per node (higher=better recall)
                "ef_construction": 200, # Controls index build quality and decides potential neighbour when new vector is inserted
            } 
        )

        # Repositories
        self.document_repository = DocumentRepository()
        self.chat_repository = ChatRepository()
        self.message_repository = MessageRepository()

        # Services
        self.document_svc = DocumentService(
            document_repository=DocumentRepository(),
            vector_store=self.vector_store,
            text_splitter=self.text_splitter,
        )

        self.llm = init_chat_model(
            model=settings.LLM_MODEL,
            model_provider=settings.LLM_PROVIDER,
            temperature=0.1,
            api_key=settings.GROQ_API_KEY
        )
        self.research_graph = build_research_graph(self.llm)

        self.ai_svc = AIService(self.llm)

        self.chat_svc = ChatService(
            chat_repository=self.chat_repository,
            message_repository=self.message_repository,
            ai_svc=self.ai_svc,
            document_svc=self.document_svc,
            research_graph=self.research_graph,
        )

def create_app_context() -> AppContext:
    """Creating an instance of app context class.

    Returns:
        AppContext: Loads app context object parsed with settings configuration
    """
    return AppContext(settings)