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
        self._session_locks: dict[str, threading.Lock] = {}

    def create(self) -> WheelSession:
        session = WheelSession(token=secrets.token_urlsafe(16), created_at=time.time())
        with self._lock:
            self._sessions[session.token] = session
        return session

    def get(self, token: str) -> WheelSession | None:
        with self._lock:
            return self._sessions.get(token)

    def lock_for(self, token: str) -> threading.Lock:
        """A per-session lock guarding that session's mutation sequences.

        Session state (entries, picks, media) lives outside the store's
        dict lock, so controllers wrap each mutate→render sequence in
        this lock: two concurrent chats (or a chat and a spin) never
        interleave halfway through a mutation. Stable per token; dropped
        by ``expire``.
        """
        with self._lock:
            lock = self._session_locks.get(token)
            if lock is None:
                lock = self._session_locks[token] = threading.Lock()
            return lock

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
                self._session_locks.pop(token, None)
            return len(expired)
