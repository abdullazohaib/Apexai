from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.config import settings
from app.core.logger import logger


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """Stores every question + all AI responses + scores + synthesized answer."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    file_context = Column(Text, nullable=True)

    # Raw responses
    response_chatgpt = Column(Text, nullable=True)
    response_gemini = Column(Text, nullable=True)
    response_claude = Column(Text, nullable=True)

    # Scores (0.0 – 1.0)
    score_chatgpt = Column(Float, nullable=True)
    score_gemini = Column(Float, nullable=True)
    score_claude = Column(Float, nullable=True)

    # Detailed breakdown stored as JSON
    breakdown_chatgpt = Column(JSON, nullable=True)
    breakdown_gemini = Column(JSON, nullable=True)
    breakdown_claude = Column(JSON, nullable=True)

    winner = Column(String(50), nullable=True)  # "chatgpt" | "gemini" | "claude"
    final_answer = Column(Text, nullable=True)
    from_cache = Column(Integer, default=0)  # 0=fresh, 1=cached

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "file_context": self.file_context,
            "responses": {
                "chatgpt": self.response_chatgpt,
                "gemini": self.response_gemini,
                "claude": self.response_claude,
            },
            "scores": {
                "chatgpt": round(self.score_chatgpt or 0, 4),
                "gemini": round(self.score_gemini or 0, 4),
                "claude": round(self.score_claude or 0, 4),
            },
            "breakdowns": {
                "chatgpt": self.breakdown_chatgpt,
                "gemini": self.breakdown_gemini,
                "claude": self.breakdown_claude,
            },
            "winner": self.winner,
            "final_answer": self.final_answer,
            "from_cache": bool(self.from_cache),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Engine & Session ──────────────────────────────────────────────────────────

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()