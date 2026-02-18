from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv 

load_dotenv()

# Configuration settings loaded from the environment/.env file.
class Settings(BaseSettings):
    """
    Automatically loads settings from various sources, such as environment variables and secrets files, with a defined order of precedence. 

    Args:
        BaseSettings (class): Automatic source loading
    """
    APP_NAME: str = "Enterprise AI Knowledge assistant"
    ENV_STATE: str = "development"
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    DB_HOST: str
    DB_PORT: int
    DB_PASSWORD: str
    DB_NAME: str
    DB_USER: str

    SECRET_KEY: str
    GROQ_API_KEY: str

    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING_V2: bool
    LANGSMITH_PROJECT: str

    # LLM Configuration, Defaulting to Groq/Llama, but can be overridden in .env
    LLM_PROVIDER: str = "groq" 
    LLM_MODEL: str 

     # Vector / NLP settings
    VECTOR_DB_PATH: str  
    CHUNK_SIZE: int = 1200          
    CHUNK_OVERLAP: int = 250       
    EMBEDDING_MODEL: str 

    vector_store_collection_name: str = "knowledge_base"

settings = Settings()
