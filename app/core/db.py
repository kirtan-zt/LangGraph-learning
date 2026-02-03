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
    """Asynchronous database session to store real time input-output query with database."""
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
            pool_recycle=3600,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
            class_=AsyncSession,
        )

    async def create_all(self) -> None:
        """Create all tables"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """FastAPI dependency."""
        async with self._session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

db = AsyncDatabaseSession()
db.init()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in db.session():
        yield session

async def init_db() -> None:
    """Explicit init hook if needed."""
    await db.create_all()


async def create_log_entry(
    session: AsyncSession,
    log: Message,
) -> Message:
    """Async session to store message logs of past conversations

    Args:
        session (AsyncSession): database instance.
        log (Message): Dictionary to store conversation data

    Returns:
        Message: Message model that stores chat id, content, timestamps, etc.
    """
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log
