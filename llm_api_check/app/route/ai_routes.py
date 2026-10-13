
import httpx

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


class PromptRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=10000,
    )


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class Performance(BaseModel):
    prompt_eval_duration_ns: int
    generation_duration_ns: int
    total_duration_ns: int
    tokens_per_second: float


class PromptResponse(BaseModel):
    model: str
    answer: str
    usage: TokenUsage
    performance: Performance


@router.post(
    "/generate",
    response_model=PromptResponse,
)
async def generate_answer(
    payload: PromptRequest,
    request: Request,
):
    ai_client = request.app.state.ai_client

    try:
        result = await ai_client.generate(payload.prompt)
        return result

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The Ollama request timed out.",
        ) from exc

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Check whether Ollama is running.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unsuccessful response.",
        ) from exc

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not parse the Ollama response.",
        ) from exc
