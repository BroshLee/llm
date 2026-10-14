
import os

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def create_database():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured"
        )

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
    )

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    return engine, session_factory


async def create_tables(engine):
    # Importing models registers their tables with Base.
    from app.models import (
        Conversation,
        ConversationMessage,
        LLMRequest,
    )

    async with engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all
        )
