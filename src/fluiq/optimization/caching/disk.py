import hashlib
import os
import pickle
import time
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union

from fluiq.optimization.caching.base import BaseCache


class DiskCache(BaseCache):
    """File-backed cache for embeddings, responses, and chunked documents.

    Each entry is a single pickle file under ``directory/`` named by SHA-256
    of the key. Suitable for medium-sized caches (10k-1M entries) that need
    to survive process restarts; for very high write throughput or shared
    multi-host caches, use a dedicated KV store.
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        default_ttl: Optional[float] = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._lock = Lock()

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{h}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with self._lock:
                with open(path, "rb") as f:
                    value, expires_at = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError):
            return None
        if expires_at is not None and time.time() > expires_at:
            try:
                path.unlink()
            except OSError:
                pass
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl is not None else None
        path = self._path(key)
        tmp = path.with_suffix(".pkl.tmp")
        with self._lock:
            with open(tmp, "wb") as f:
                pickle.dump((value, expires_at), f)
            os.replace(tmp, path)

    def delete(self, key: str) -> bool:
        try:
            self._path(key).unlink()
            return True
        except FileNotFoundError:
            return False

    def clear(self) -> None:
        with self._lock:
            for p in self.directory.glob("*.pkl"):
                try:
                    p.unlink()
                except OSError:
                    pass
