import os

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import text

load_dotenv()

from app.database import create_database
from app.logging_config import configure_logging
from app.middleware import RequestLoggingMiddleware
from app.route.ai_routes import router as ai_router
from app.security import validate_security_config
from app.services.ai_client import AIClient
from app.services.response_cache import ResponseCache

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_security_config()

    engine, session_factory = create_database()

    app.state.engine = engine
    app.state.session_factory = session_factory

    try:
        # Fail fast if PostgreSQL is unavailable.
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        app.state.ai_client = AIClient()
        app.state.response_cache = ResponseCache(
            sqlite_path=os.getenv(
                "LLM_CACHE_SQLITE_PATH",
                "./data/response_cache.sqlite3",
            ),
            qdrant_path=os.getenv(
                "QDRANT_LOCAL_PATH",
                "./data/qdrant",
            ),
            ttl_seconds=int(
                os.getenv("LLM_CACHE_TTL_SECONDS", "604800")
            ),
            similarity_threshold=float(
                os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92")
            ),
            cache_version=os.getenv("LLM_CACHE_VERSION", "v1"),
            embedding_model=os.getenv(
                "OLLAMA_EMBED_MODEL",
                "nomic-embed-text",
            ),
        )
        
        yield

    finally:
        if hasattr(app.state, "response_cache"):
            app.state.response_cache.close()
        if hasattr(app.state, "ai_client"):
            await app.state.ai_client.close()

        await engine.dispose()


app = FastAPI(
    title="Ollama LLM API",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(ai_router)


@app.get("/health")
async def health():
    # Liveness check only; it does not expose internal configuration.
    return {"status": "ok"}
