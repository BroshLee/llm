
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

        yield

    finally:
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
