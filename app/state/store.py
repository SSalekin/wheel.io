import secrets
import threading
import time

from app.models.session import WheelSession


class InMemoryStateStore:
    """Per-page session store; nothing survives process restart or TTL expiry."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, WheelSession] = {}
        self._lock = threading.Lock()

    def create(self) -> WheelSession:
        session = WheelSession(token=secrets.token_urlsafe(16), created_at=time.time())
        with self._lock:
            self._sessions[session.token] = session
        return session

    def get(self, token: str) -> WheelSession | None:
        with self._lock:
            return self._sessions.get(token)

    def expire(self, now: float | None = None) -> int:
        if now is None:
            now = time.time()
        with self._lock:
            expired = [
                token
                for token, session in self._sessions.items()
                if now - session.created_at > self.ttl_seconds
            ]
            for token in expired:
                del self._sessions[token]
            return len(expired)
