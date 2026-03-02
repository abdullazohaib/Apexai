import time
import hashlib
import json
from typing import Any, Optional
from threading import Lock
from app.config import settings
from app.core.logger import logger


class InMemoryCache:
    """Thread-safe in-memory cache with TTL support. Swap this for Redis in production."""

    def __init__(self, ttl: int = settings.CACHE_TTL):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl
        self._lock = Lock()

    def _make_key(self, raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hashed = self._make_key(key)
        with self._lock:
            entry = self._store.get(hashed)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[hashed]
                logger.debug(f"Cache MISS (expired): {key[:50]}")
                return None
            logger.debug(f"Cache HIT: {key[:50]}")
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        hashed = self._make_key(key)
        expires_at = time.time() + (ttl or self._ttl)
        with self._lock:
            self._store[hashed] = (value, expires_at)
        logger.debug(f"Cache SET: {key[:50]}")

    def delete(self, key: str) -> None:
        hashed = self._make_key(key)
        with self._lock:
            self._store.pop(hashed, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        logger.info("Cache cleared")

    def stats(self) -> dict:
        with self._lock:
            total = len(self._store)
            now = time.time()
            active = sum(1 for _, (_, exp) in self._store.items() if exp > now)
        return {"total_keys": total, "active_keys": active, "expired_keys": total - active}


def make_cache_key(question: str, responses: dict) -> str:
    payload = {"q": question.lower().strip(), "r": responses}
    return json.dumps(payload, sort_keys=True)


# Singleton cache instance
cache = InMemoryCache()