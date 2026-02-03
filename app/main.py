from fastapi import FastAPI
from app.core.config import settings
from app.routes import chat, document, logs
from app.core.db import init_db

def create_application() -> FastAPI:
    app = FastAPI(
    title="Content Research Agent built with LangGraph",
    )

    app.include_router(
        document.router,
        )
    app.include_router(
        chat.router,
        )
    app.include_router(
        logs.router,
        )

    return app


app = create_application()

@app.on_event("startup")
async def startup():
    await init_db()

@app.on_event("shutdown")
async def on_shutdown():
    pass
