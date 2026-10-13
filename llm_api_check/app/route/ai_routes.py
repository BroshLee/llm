
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import httpx

router = APIRouter(prefix="/ai", tags=["AI"])


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)


class TokenUsage(BaseModel):
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_second: float


class PromptResponse(BaseModel):
    answer: str
    usage: TokenUsage


@router.post("/generate", response_model=PromptResponse)
async def generate_answer(
    payload: PromptRequest,
    request: Request,
):
    ai_client = request.app.state.ai_client

    try:
        result = await ai_client.generate(payload.prompt)
        return PromptResponse(
            answer=result.answer,
            usage=TokenUsage(
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.prompt_tokens + result.output_tokens,
                tokens_per_second=result.tokens_per_second,
            ),
        )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The LLM request timed out.",
        ) from exc

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unsuccessful response.",
        ) from exc
