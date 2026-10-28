
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal

import httpx
import time

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.models import (
    Conversation,
    ConversationMessage,
    LLMRequest,
)
from app.security import require_api_key

from app.schemas.ai_schemas import (
    SkillExtractionRequest,
    SkillExtractionResult,
    StructuredJSONResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["AI"],
    dependencies=[Depends(require_api_key)],
)


# ---------- Input validation ----------

# "auto" picks the math model when the text looks like a math question.
Task = Literal["auto", "math", "general"]


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    task: Task = "auto"

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Prompt cannot be empty or whitespace.")

        return value


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: int | None = Field(default=None, gt=0)
    task: Task = "auto"

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be empty or whitespace.")

        return value


# ---------- Response schemas ----------

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
    cache_hit: bool = False


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


# ---------- Serialization and persistence ----------

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


@contextmanager
def ollama_errors(request: Request):
    try:
        yield

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
        logger.warning(
            "ollama_http_error",
            extra={
                "status_code": exc.response.status_code,
                "request_id": getattr(
                    request.state, "request_id", None
                ),
            },
        )
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unsuccessful response.",
        ) from exc

    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not parse the Ollama response.",
        ) from exc


async def call_ollama(
    request: Request,
    messages: list[dict],
    model: str,
) -> dict:
    with ollama_errors(request):
        return await request.app.state.ai_client.chat(messages, model)


async def open_ollama_stream(
    request: Request,
    messages: list[dict],
    model: str,
) -> httpx.Response:
    with ollama_errors(request):
        return await request.app.state.ai_client.open_chat_stream(
            messages,
            model,
        )


def stream_answer(
    request: Request,
    response: httpx.Response,
    save,
    headers: dict | None = None,
) -> StreamingResponse:
    async def body():
        result = None

        try:
            async for item in request.app.state.ai_client.iter_chat_stream(
                response
            ):
                if isinstance(item, str):
                    yield item
                else:
                    result = item

        except (httpx.HTTPError, ValueError):
            # Status 200 is already sent, so report the failure in-band.
            logger.exception(
                "ollama_stream_failed",
                extra={"request_id": request.state.request_id},
            )
            yield "\n\n[error: the answer was interrupted]"
            return

        if result is None:
            return

        try:
            await save(result)
        except HTTPException:
            # Already logged by the save helper; the answer was delivered.
            pass

    return StreamingResponse(
        body(),
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


async def save_generation(
    request: Request,
    prompt: str,
    result: dict,
) -> int:
    try:
        async with request.app.state.session_factory() as session:
            async with session.begin():
                record = await save_llm_request(
                    session,
                    result,
                    prompt,
                )
                await session.flush()
                return record.id

    except SQLAlchemyError as exc:
        logger.exception(
            "llm_request_persistence_failed",
            extra={
                "request_id": request.state.request_id,
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Could not save the LLM request.",
        ) from exc


async def load_conversation(
    request: Request,
    payload: ChatRequest,
) -> tuple[int, list[dict]]:
    session_factory = request.app.state.session_factory
    conversation_id = payload.conversation_id

    # Create a conversation if this is the first message.
    try:
        async with session_factory() as session:
            if conversation_id is None:
                async with session.begin():
                    conversation = Conversation(
                        title=payload.message[:200],
                    )
                    session.add(conversation)
                    await session.flush()
                    conversation_id = conversation.id

            else:
                conversation = await session.get(
                    Conversation, conversation_id
                )

                if conversation is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Conversation not found.",
                    )

            # Only the latest 20 messages are sent as context.
            history_query = (
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id
                    == conversation_id
                )
                .order_by(ConversationMessage.id.desc())
                .limit(20)
            )

            history_rows = (
                await session.scalars(history_query)
            ).all()

            history = [
                {"role": row.role, "content": row.content}
                for row in reversed(history_rows)
            ]

    except SQLAlchemyError as exc:
        logger.exception(
            "conversation_load_failed",
            extra={"request_id": request.state.request_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Could not load conversation history.",
        ) from exc

    return conversation_id, history


async def save_chat_turn(
    request: Request,
    conversation_id: int,
    message: str,
    result: dict,
) -> int:
    # Persist both messages and the usage record in one transaction.
    try:
        async with request.app.state.session_factory() as session:
            async with session.begin():
                conversation = await session.get(
                    Conversation, conversation_id
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
                        content=message,
                    ),
                    ConversationMessage(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=result["answer"],
                    ),
                ])

                conversation.updated_at = datetime.now(timezone.utc)

                record = await save_llm_request(
                    session,
                    result,
                    message,
                    conversation_id,
                )

                await session.flush()
                return record.id

    except SQLAlchemyError as exc:
        logger.exception(
            "conversation_persistence_failed",
            extra={
                "request_id": request.state.request_id,
                "conversation_id": conversation_id,
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Could not save the conversation.",
        ) from exc


# ---------- Single-prompt generation ----------

@router.post("/generate", response_model=PromptResponse)
async def generate_answer(
    payload: PromptRequest,
    request: Request,
):
    ai_client = request.app.state.ai_client
    response_cache = request.app.state.response_cache
    model = ai_client.model_for(payload.prompt, payload.task)

    # Measure cache lookup time separately from LLM generation.
    cache_started = time.perf_counter_ns()

    cached_result, query_vector = await response_cache.lookup(
        prompt=payload.prompt,
        model=model,
        ai_client=ai_client,
    )

    if cached_result is not None:
        # Exact or semantic cache hit: do not call the chat model.
        elapsed_ns = time.perf_counter_ns() - cache_started

        result = response_cache.make_cache_hit(
            cached_result,
            elapsed_ns,
        )
        cache_hit = True

    else:
        # Cache miss: use the existing Ollama call.
        result = await call_ollama(
            request,
            [
                {
                    "role": "user",
                    "content": payload.prompt,
                }
            ],
            model,
        )

        cache_hit = False

        # Save the answer for future exact and semantic lookups.
        await response_cache.store(
            prompt=payload.prompt,
            model=model,
            result=result,
            ai_client=ai_client,
            vector=query_vector,
        )

    # Keep the existing PostgreSQL history and usage persistence.
    record_id = await save_generation(
        request,
        payload.prompt,
        result,
    )

    return {
        **result,
        "id": record_id,
        "cache_hit": cache_hit,
    }

@router.post("/generate/stream")
async def generate_answer_stream(
    payload: PromptRequest,
    request: Request,
):
    model = request.app.state.ai_client.model_for(
        payload.prompt,
        payload.task,
    )

    response = await open_ollama_stream(
        request,
        [{"role": "user", "content": payload.prompt}],
        model,
    )

    async def save(result: dict) -> None:
        await save_generation(request, payload.prompt, result)

    return stream_answer(request, response, save)


# ---------- Conversation memory ----------

@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    conversation_id, history = await load_conversation(request, payload)

    # Routed per message, so one conversation can mix models.
    model = request.app.state.ai_client.model_for(
        payload.message,
        payload.task,
    )

    result = await call_ollama(
        request,
        [*history, {"role": "user", "content": payload.message}],
        model,
    )

    record_id = await save_chat_turn(
        request,
        conversation_id,
        payload.message,
        result,
    )

    return {
        **result,
        "id": record_id,
        "conversation_id": conversation_id,
    }


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    conversation_id, history = await load_conversation(request, payload)

    model = request.app.state.ai_client.model_for(
        payload.message,
        payload.task,
    )

    response = await open_ollama_stream(
        request,
        [*history, {"role": "user", "content": payload.message}],
        model,
    )

    async def save(result: dict) -> None:
        await save_chat_turn(
            request,
            conversation_id,
            payload.message,
            result,
        )

    # The body is plain text, so the conversation id travels in a header.
    return stream_answer(
        request,
        response,
        save,
        headers={"X-Conversation-ID": str(conversation_id)},
    )


# ---------- Paginated request history ----------

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


# ---------- Retrieve one request ----------

@router.get("/history/{request_id}", response_model=HistoryItem)
async def get_request_detail(request_id: int, request: Request):
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


# ---------- Token usage analytics ----------

@router.get("/usage", response_model=TokenUsageSummary)
async def get_token_usage(request: Request):
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
        async with request.app.state.session_factory() as session:
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


# ---------- Conversation messages ----------

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
                Conversation, conversation_id
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
                .limit(500)
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

# ---------- generate json messages ----------
@router.post(
    "/generate-json",
    response_model=StructuredJSONResponse,
)
async def generate_json(
    payload: SkillExtractionRequest,
    request: Request,
):
    prompt = f"""
Extract information from the following text.

Return:
- name: person's name
- role: person's professional role
- skills: list of technical skills

If a value cannot be identified, use an empty string
for name or role, and an empty list for skills.

Text:
{payload.text}
"""

    schema = SkillExtractionResult.model_json_schema()

    try:
        result = await request.app.state.ai_client.generate_json(
            prompt=prompt,
            schema=schema,
        )

        # Validate the JSON returned by the LLM.
        validated_data = SkillExtractionResult.model_validate(
            result["data"]
        )

        return {
            "model": result["model"],
            "data": validated_data,
            "usage": result["usage"],
            "performance": result["performance"],
        }

    except ValidationError as exc:
        logging.exception("Invalid structured JSON from Ollama")
        raise HTTPException(
            status_code=502,
            detail="The LLM returned data that did not match the expected schema.",
        ) from exc

    except ValueError as exc:
        logging.exception("Failed to parse JSON returned by Ollama")
        raise HTTPException(
            status_code=502,
            detail="The LLM returned invalid JSON.",
        ) from exc

    except httpx.HTTPError as exc:
        logging.exception("Ollama request failed")
        raise HTTPException(
            status_code=502,
            detail="The Ollama service request failed.",
        ) from exc