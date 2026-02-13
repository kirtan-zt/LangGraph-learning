from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from app.core.config import settings
from app.core.base import Base
from app.models.logs import Message
import urllib.parse

# Encode password safely
password_encoded = urllib.parse.quote_plus(settings.DB_PASSWORD)

class AsyncDatabaseSession:
    """
    Manages the asynchronous SQLAlchemy engine and session factory.

    Provides a centralized interface for database initialization and 
    dependency injection for asynchronous session management.
    """
    def __init__(self):
        self._engine = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    def init(self) -> None:
        DATABASE_URL = (
            f"postgresql+asyncpg://{settings.DB_USER}:"
            f"{password_encoded}@"
            f"{settings.DB_HOST}:{settings.DB_PORT}/"
            f"{settings.DB_NAME}"
        )

        self._engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            future=True,
            pool_recycle=3600, # Prevent stale connections by recycling hourly
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
            class_=AsyncSession,
        )

    async def create_all(self) -> None:
        """Synchronizes the database schema with the current SQLAlchemy metadata."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Generates a new scoped asynchronous session.

        Yields:
            AsyncSession: An active database session.
        """
        async with self._session_factory() as session:
            try:
                yield session
            finally:
                # Ensure session cleanup regardless of execution outcome
                await session.close()

# Global database manager instance
db = AsyncDatabaseSession()
db.init()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency provider for FastAPI route handlers.
    
    Yields:
        AsyncSession: Managed database session for request lifecycle.
    """
    async for session in db.session():
        yield session

async def init_db() -> None:
    """
    Application-level hook to trigger schema creation.
    """
    await db.create_all()


async def create_log_entry(
    session: AsyncSession,
    log: Message,
) -> Message:
    """
    Persists a new Message log entry to the database.

    Args:
        session: Active SQLAlchemy asynchronous session.
        log: The Message model instance to be persisted.

    Returns:
        Message: The persisted model instance with database-generated attributes.
    """
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log
