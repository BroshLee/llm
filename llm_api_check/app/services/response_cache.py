
import hashlib
import json
import logging
import re
import sqlite3
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Persistent exact cache + Qdrant semantic cache.

    Exact cache:
        SQLite lookup; no embedding or LLM call on a hit.

    Semantic cache:
        Ollama embedding + Qdrant similarity search.
    """

    def __init__(
        self,
        sqlite_path: str = "./data/response_cache.sqlite3",
        qdrant_path: str = "./data/qdrant",
        ttl_seconds: int = 7 * 24 * 60 * 60,
        similarity_threshold: float = 0.92,
        cache_version: str = "v1",
        embedding_model: str = "nomic-embed-text",
    ):
        self.sqlite_path = str(Path(sqlite_path).resolve())
        self.qdrant_path = str(Path(qdrant_path).resolve())
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.cache_version = cache_version
        self.embedding_model = embedding_model

        Path(self.sqlite_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(self.qdrant_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        # Use a different collection name for different embedding models.
        safe_model = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            embedding_model,
        )
        self.collection_name = f"response_cache_{safe_model}"

        # Embedded local Qdrant. No Docker or server required.
        self.qdrant = QdrantClient(path=self.qdrant_path)

        self._initialize_sqlite()

        # These prompts are likely to require fresh information.
        self.dynamic_prompt_pattern = re.compile(
            r"\b("
            r"today|tomorrow|yesterday|right now|currently|current|"
            r"latest|recent|news|weather|stock price|share price|"
            r"exchange rate|forecast|price of|time in"
            r")\b",
            re.IGNORECASE,
        )

        # Avoid caching questions that depend on private or changing context.
        self.context_prompt_pattern = re.compile(
            r"\b("
            r"previous conversation|previous message|above conversation|"
            r"my account|my file|my document|this document|"
            r"based on the above|based on my data"
            r")\b",
            re.IGNORECASE,
        )

    def _initialize_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_response_cache_expiry
                ON response_cache(expires_at)
                """
            )

    def is_cacheable(self, prompt: str) -> bool:
        prompt = prompt.strip()

        if not prompt:
            return False

        if self.dynamic_prompt_pattern.search(prompt):
            return False

        if self.context_prompt_pattern.search(prompt):
            return False

        return True

    def _normalize_prompt(self, prompt: str) -> str:
        return " ".join(prompt.split()).casefold()

    def _make_key(self, prompt: str, model: str) -> str:
        # Change cache_version when prompt/model behavior changes.
        raw = "|".join(
            [
                self.cache_version,
                model,
                self._normalize_prompt(prompt),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_exact(self, key: str) -> dict | None:
        now = time.time()

        with sqlite3.connect(self.sqlite_path) as connection:
            row = connection.execute(
                """
                SELECT result_json, expires_at
                FROM response_cache
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()

            if row is None:
                return None

            result_json, expires_at = row

            if expires_at <= now:
                connection.execute(
                    "DELETE FROM response_cache WHERE cache_key = ?",
                    (key,),
                )
                return None

            return json.loads(result_json)

    def _save_exact(
        self,
        key: str,
        prompt: str,
        model: str,
        result: dict,
    ) -> None:
        expires_at = time.time() + self.ttl_seconds

        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO response_cache (
                    cache_key,
                    prompt,
                    model,
                    result_json,
                    expires_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    prompt = excluded.prompt,
                    model = excluded.model,
                    result_json = excluded.result_json,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    prompt,
                    model,
                    json.dumps(result),
                    expires_at,
                ),
            )

    def _ensure_collection(self, vector_size: int) -> None:
        if self.qdrant.collection_exists(self.collection_name):
            return

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    async def lookup(
        self,
        prompt: str,
        model: str,
        ai_client,
    ) -> tuple[dict | None, list[float] | None]:
        """
        Returns (cached_result, query_vector).

        A None result means the caller should invoke the LLM.
        The vector is returned so the caller can reuse it when storing
        a cache miss instead of embedding the same question twice.
        """
        if not self.is_cacheable(prompt):
            return None, None

        key = self._make_key(prompt, model)

        # 1. Exact cache: no embedding model call needed.
        try:
            exact_result = self._get_exact(key)
            if exact_result is not None:
                logger.info("response_cache_exact_hit")
                return exact_result, None
        except (sqlite3.Error, ValueError, TypeError):
            logger.exception("response_cache_exact_lookup_failed")

        # 2. If there are no vectors yet, avoid an unnecessary embedding call.
        try:
            if not self.qdrant.collection_exists(
                self.collection_name
            ):
                return None, None

            query_vector = await ai_client.embed(prompt)

            query_result = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="model",
                            match=models.MatchValue(value=model),
                        ),
                        models.FieldCondition(
                            key="cache_version",
                            match=models.MatchValue(
                                value=self.cache_version
                            ),
                        ),
                    ]
                ),
                limit=5,
                with_payload=True,
            )

            now = time.time()

            for point in query_result.points:
                payload = point.payload or {}

                if point.score < self.similarity_threshold:
                    continue

                if payload.get("expires_at", 0) <= now:
                    continue

                cached_result = payload.get("result")
                if not isinstance(cached_result, dict):
                    continue

                # Remember this wording in the exact cache too.
                self._save_exact(
                    key,
                    prompt,
                    model,
                    cached_result,
                )

                logger.info(
                    "response_cache_semantic_hit",
                    extra={"similarity_score": point.score},
                )
                return cached_result, query_vector

            return None, query_vector

        except Exception:
            # A cache failure must not stop a normal LLM request.
            logger.exception("response_cache_semantic_lookup_failed")
            return None, None

    async def store(
        self,
        prompt: str,
        model: str,
        result: dict,
        ai_client,
        vector: list[float] | None = None,
    ) -> None:
        if not self.is_cacheable(prompt):
            return

        key = self._make_key(prompt, model)

        # Store the exact result first. This still works if Qdrant or
        # the embedding model is temporarily unavailable.
        try:
            self._save_exact(key, prompt, model, result)
        except (sqlite3.Error, TypeError, ValueError):
            logger.exception("response_cache_exact_store_failed")

        try:
            if vector is None:
                vector = await ai_client.embed(prompt)

            self._ensure_collection(len(vector))

            payload = {
                "prompt": prompt,
                "model": model,
                "cache_version": self.cache_version,
                "expires_at": time.time() + self.ttl_seconds,
                "result": result,
            }

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, key)
            )

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
                wait=True,
            )

        except Exception:
            # Semantic caching is an optimization, not a dependency
            # for returning a valid LLM answer.
            logger.exception("response_cache_vector_store_failed")

    @staticmethod
    def make_cache_hit(
        cached_result: dict,
        elapsed_ns: int,
    ) -> dict:
        """
        Return the cached answer but report zero newly generated tokens.
        The duration represents this cache lookup, not LLM generation.
        """
        result = dict(cached_result)

        result["usage"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        result["performance"] = {
            "prompt_eval_duration_ns": 0,
            "generation_duration_ns": 0,
            "total_duration_ns": elapsed_ns,
            "tokens_per_second": 0.0,
        }

        result["cache_hit"] = True
        return result

    def close(self) -> None:
        self.qdrant.close()
