
import httpx

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.models import LLMRequest


router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


# ---------- Request and response schemas ----------

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


class HistoryItem(BaseModel):
    id: int
    prompt: str
    answer: str
    model: str
    usage: TokenUsage
    performance: Performance
    created_at: datetime


class PaginatedHistory(BaseModel):
    items: list[HistoryItem]
    total_records: int
    page: int
    limit: int
    total_pages: int


class TokenUsageSummary(BaseModel):
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    average_tokens_per_second: float


# ---------- Shared database response formatter ----------

def serialize_request(row: LLMRequest) -> dict:
    return {
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
            "prompt_eval_duration_ns": row.prompt_eval_duration_ns,
            "generation_duration_ns": row.generation_duration_ns,
            "total_duration_ns": row.total_duration_ns,
            "tokens_per_second": row.tokens_per_second,
        },
        "created_at": row.created_at,
    }


# ---------- Generate an LLM response ----------

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

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "LLM response generated, but saving to the database failed."
            ),
        ) from exc

    # Step 3: Return the response and its database ID.
    return {
        **result,
        "id": record_id,
    }


# ---------- 1. Paginated request history ----------

@router.get(
    "/history",
    response_model=PaginatedHistory,
)
async def get_history(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    session_factory = request.app.state.session_factory

    offset = (page - 1) * limit

    try:
        async with session_factory() as session:
            # Count all matching records.
            count_query = select(func.count()).select_from(LLMRequest)
            total_records = await session.scalar(count_query)
            total_records = total_records or 0

            # Fetch only the requested page.
            history_query = (
                select(LLMRequest)
                .order_by(LLMRequest.id.desc())
                .offset(offset)
                .limit(limit)
            )

            result = await session.scalars(history_query)
            rows = result.all()

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve request history.",
        ) from exc

    total_pages = (
        (total_records + limit - 1) // limit
        if total_records
        else 0
    )

    return {
        "items": [serialize_request(row) for row in rows],
        "total_records": total_records,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }


# ---------- 2. Retrieve one request by ID ----------

@router.get(
    "/history/{request_id}",
    response_model=HistoryItem,
)
async def get_request_detail(
    request_id: int,
    request: Request,
):
    session_factory = request.app.state.session_factory

    try:
        async with session_factory() as session:
            query = select(LLMRequest).where(
                LLMRequest.id == request_id
            )

            row = await session.scalar(query)

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve the requested record.",
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"LLM request with ID {request_id} was not found.",
        )

    return serialize_request(row)


# ---------- 3. Aggregate token usage in PostgreSQL ----------

@router.get(
    "/usage",
    response_model=TokenUsageSummary,
)
async def get_token_usage(request: Request):
    session_factory = request.app.state.session_factory

    query = select(
        func.count(LLMRequest.id).label("total_requests"),
        func.coalesce(
            func.sum(LLMRequest.input_tokens), 0
        ).label("total_input_tokens"),
        func.coalesce(
            func.sum(LLMRequest.output_tokens), 0
        ).label("total_output_tokens"),
        func.coalesce(
            func.sum(LLMRequest.total_tokens), 0
        ).label("total_tokens"),
        func.coalesce(
            func.avg(LLMRequest.tokens_per_second), 0.0
        ).label("average_tokens_per_second"),
    )

    try:
        async with session_factory() as session:
            result = (await session.execute(query)).mappings().one()

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not calculate token usage.",
        ) from exc

    return {
        "total_requests": result["total_requests"],
        "total_input_tokens": result["total_input_tokens"],
        "total_output_tokens": result["total_output_tokens"],
        "total_tokens": result["total_tokens"],
        "average_tokens_per_second": round(
            float(result["average_tokens_per_second"]), 2
        ),
    }
