
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class LLMRequest(Base):
    __tablename__ = "llm_requests"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    prompt_eval_duration_ns: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    generation_duration_ns: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    total_duration_ns: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    tokens_per_second: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
