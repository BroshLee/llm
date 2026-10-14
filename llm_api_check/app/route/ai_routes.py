
from datetime import datetime

import httpx

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.models import (
    Conversation,
    ConversationMessage,
    LLMRequest,
)


router = APIRouter(prefix="/ai", tags=["AI"])


# ---------- Schemas ----------

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: int | None = Field(default=None, gt=0)


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


class ChatResponse(PromptResponse):
    conversation_id: int


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


class ConversationMessageItem(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime


# ---------- Helpers ----------

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


async def save_llm_request(
    session,
    result: dict,
    prompt: str,
    conversation_id: int | None = None,
) -> LLMRequest:
    record = LLMRequest(
        conversation_id=conversation_id,
        prompt=prompt,
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

    session.add(record)
    await session.flush()

    return record


# ---------- Existing endpoint: single-prompt generation ----------

@router.post("/generate", response_model=PromptResponse)
async def generate_answer(
    payload: PromptRequest,
    request: Request,
):
    ai_client = request.app.state.ai_client
    session_factory = request.app.state.session_factory

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

    try:
        async with session_factory() as session:
            async with session.begin():
                record = await save_llm_request(
                    session,
                    result,
                    payload.prompt,
                )
                await session.flush()
                record_id = record.id

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save the LLM request.",
        ) from exc

    return {
        **result,
        "id": record_id,
    }


# ---------- New endpoint: conversation memory ----------

@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
):
    ai_client = request.app.state.ai_client
    session_factory = request.app.state.session_factory

    conversation_id = payload.conversation_id
    messages = []

    # 1. Load an existing conversation or create a new one.
    try:
        async with session_factory() as session:
            async with session.begin():
                if conversation_id is None:
                    conversation = Conversation(
                        title=payload.message[:200],
                    )
                    session.add(conversation)
                    await session.flush()
                    conversation_id = conversation.id

                else:
                    conversation = await session.get(
                        Conversation,
                        conversation_id,
                    )

                    if conversation is None:
                        raise HTTPException(
                            status_code=404,
                            detail="Conversation not found.",
                        )

                # Fetch previous messages in chronological order.
                history_query = (
                    select(ConversationMessage)
                    .where(
                        ConversationMessage.conversation_id
                        == conversation_id
                    )
                    .order_by(ConversationMessage.id.asc())
                )

                history_result = await session.scalars(
                    history_query
                )
                history = history_result.all()

                messages = [
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in history
                ]

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not load conversation history.",
        ) from exc

    # 2. Append the new message to the in-memory context.
    messages.append({
        "role": "user",
        "content": payload.message,
    })

    # 3. Send the full context to Ollama.
    try:
        result = await ai_client.chat(messages)

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

    # 4. Persist both messages and token metrics atomically.
    try:
        async with session_factory() as session:
            async with session.begin():
                conversation = await session.get(
                    Conversation,
                    conversation_id,
                )

                if conversation is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Conversation no longer exists.",
                    )

                session.add_all([
                    ConversationMessage(
                        conversation_id=conversation_id,
                        role="user",
                        content=payload.message,
                    ),
                    ConversationMessage(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=result["answer"],
                    ),
                ])

                conversation.updated_at = func.now()

                record = await save_llm_request(
                    session,
                    result,
                    payload.message,
                    conversation_id,
                )

                await session.flush()
                record_id = record.id

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not save the conversation.",
        ) from exc

    return {
        **result,
        "id": record_id,
        "conversation_id": conversation_id,
    }


# ---------- Existing endpoint: paginated request history ----------

@router.get("/history", response_model=PaginatedHistory)
async def get_history(
    request: Request,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    session_factory = request.app.state.session_factory
    offset = (page - 1) * limit

    try:
        async with session_factory() as session:
            total_records = await session.scalar(
                select(func.count()).select_from(LLMRequest)
            )
            total_records = total_records or 0

            query = (
                select(LLMRequest)
                .order_by(LLMRequest.id.desc())
                .offset(offset)
                .limit(limit)
            )

            rows = (await session.scalars(query)).all()

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


# ---------- Existing endpoint: one request by ID ----------

@router.get("/history/{request_id}", response_model=HistoryItem)
async def get_request_detail(
    request_id: int,
    request: Request,
):
    session_factory = request.app.state.session_factory

    try:
        async with session_factory() as session:
            row = await session.scalar(
                select(LLMRequest).where(
                    LLMRequest.id == request_id
                )
            )

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve the request.",
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Request not found.",
        )

    return serialize_request(row)


# ---------- Existing endpoint: aggregate token usage ----------

@router.get("/usage", response_model=TokenUsageSummary)
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
            result = (
                await session.execute(query)
            ).mappings().one()

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


# ---------- New endpoint: read conversation messages ----------

@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[ConversationMessageItem],
)
async def get_conversation_messages(
    conversation_id: int,
    request: Request,
):
    session_factory = request.app.state.session_factory

    try:
        async with session_factory() as session:
            conversation = await session.get(
                Conversation,
                conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

            query = (
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id
                    == conversation_id
                )
                .order_by(ConversationMessage.id.asc())
            )

            rows = (await session.scalars(query)).all()

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve conversation messages.",
        ) from exc

    return [
        {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "role": row.role,
            "content": row.content,
            "created_at": row.created_at,
        }
        for row in rows
    ]
