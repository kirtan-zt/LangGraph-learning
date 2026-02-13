from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.routes import chat, document, logs
from app.core.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    
    Handles asynchronous startup tasks, such as database initialization, 
    and ensures clean resource teardown during shutdown.
    """
    # Startup: Initialize database schema and connection pools
    await init_db()
    yield
    # Shutdown: Logic for closing connections or cleaning up resources goes here
    pass

def create_application() -> FastAPI:
    """
    Factory function to initialize and configure the FastAPI application instance.
    """
    app = FastAPI(
        title="Content Research Agent",
        description="A RAG-powered research assistant built with LangGraph and FastAPI.",
        version="1.0.0",
        lifespan=lifespan
    )

    # Register API routers with distinct resource tags
    app.include_router(document.router)
    app.include_router(chat.router)
    app.include_router(logs.router)

    return app

# Main application instance 
app = create_application()