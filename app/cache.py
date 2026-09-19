"""Caching module for queries and responses."""

from typing import Any, Optional


class InMemoryCache:
    """In-memory key-value cache."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()


cache = InMemoryCache()
