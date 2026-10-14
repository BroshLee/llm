
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from app.database import create_database, create_tables
from app.route.ai_routes import router as ai_router
from app.services.ai_client import AIClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, session_factory = create_database()

    app.state.engine = engine
    app.state.session_factory = session_factory

    try:
        await create_tables(engine)

        app.state.ai_client = AIClient()

        yield

    finally:
        if hasattr(app.state, "ai_client"):
            await app.state.ai_client.close()

        await engine.dispose()


app = FastAPI(
    title="Ollama LLM API",
    lifespan=lifespan,
)

app.include_router(ai_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
