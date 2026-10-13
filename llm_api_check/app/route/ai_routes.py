
import httpx

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LLMRequest


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
    id: int
    model: str
    answer: str
    usage: TokenUsage
    performance: Performance


async def get_db(request: Request):
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        yield session


@router.post(
    "/generate",
    response_model=PromptResponse,
)
async def generate_answer(
    payload: PromptRequest,
    request: Request,
):
    ai_client = request.app.state.ai_client
    session_factory = request.app.state.session_factory

    # Step 1: Generate the LLM response.
    try:
        result = await ai_client.generate(payload.prompt)

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="The Ollama request timed out.",
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

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not parse the Ollama response.",
        ) from exc

    # Step 2: Save the response and metrics to PostgreSQL.
    try:
        async with session_factory() as session:
            db_record = LLMRequest(
                prompt=payload.prompt,
                answer=result["answer"],
                model=result["model"],
                input_tokens=result["usage"]["input_tokens"],
                output_tokens=result["usage"]["output_tokens"],
                total_tokens=result["usage"]["total_tokens"],
                prompt_eval_duration_ns=(
                    result["performance"]["prompt_eval_duration_ns"]
                ),
                generation_duration_ns=(
                    result["performance"]["generation_duration_ns"]
                ),
                total_duration_ns=(
                    result["performance"]["total_duration_ns"]
                ),
                tokens_per_second=(
                    result["performance"]["tokens_per_second"]
                ),
            )

            session.add(db_record)
            await session.commit()
            await session.refresh(db_record)

            record_id = db_record.id

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="LLM response generated, but saving to the database failed.",
        ) from exc

    # Step 3: Return the response and its database ID.
    return {
        **result,
        "id": record_id,
    }


@router.get("/history")
async def get_history(
    request: Request,
    limit: int = 20,
):
    if not 1 <= limit <= 100:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 100.",
        )

    session_factory = request.app.state.session_factory

    from sqlalchemy import select

    async with session_factory() as session:
        query = (
            select(LLMRequest)
            .order_by(LLMRequest.id.desc())
            .limit(limit)
        )

        rows = (await session.scalars(query)).all()

        return [
            {
                "id": row.id,
                "prompt": row.prompt,
                "answer": row.answer,
                "model": row.model,
                "usage": {
                    "input_tokens": row.input_tokens,
                    "output_tokens": row.output_tokens,
                    "total_tokens": row.total_tokens,
                },
                "performance": {
                    "prompt_eval_duration_ns": (
                        row.prompt_eval_duration_ns
                    ),
                    "generation_duration_ns": (
                        row.generation_duration_ns
                    ),
                    "total_duration_ns": row.total_duration_ns,
                    "tokens_per_second": row.tokens_per_second,
                },
                "created_at": row.created_at,
            }
            for row in rows
        ]
