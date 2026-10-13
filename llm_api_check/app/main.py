
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.route.ai_routes import router as ai_router
from app.services.ai_client import AIClient


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ai_client = AIClient()

    try:
        yield
    finally:
        await app.state.ai_client.close()


app = FastAPI(
    title="Ollama LLM API",
    lifespan=lifespan,
)

app.include_router(ai_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }
