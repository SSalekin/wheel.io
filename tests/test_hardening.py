"""Task 14: app-level exception handlers and the lifespan TTL sweep."""

import asyncio
import logging
import re
import time

from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.services.errors import UserFacingError

GENERIC_ERROR = "Something went wrong."


def _page_token(client) -> str:
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def _boom(*args, **kwargs):
    raise RuntimeError("unexpected failure")


def _user_facing(*args, **kwargs):
    raise UserFacingError("custom notice")


def test_unexpected_exception_bubbles(monkeypatch, caplog):
    """Anything unhandled becomes the generic bubble — HTTP 200, no stack
    trace in the UI, but the traceback still reaches the logs."""
    monkeypatch.setattr("app.controllers.chat.interpret", _boom)

    with TestClient(app, raise_server_exceptions=False) as client:
        token = _page_token(client)
        with caplog.at_level(logging.ERROR):
            response = client.post("/chat", data={"token": token, "message": "Ana, Bob"})

    assert response.status_code == 200
    assert GENERIC_ERROR in response.text
    assert "Traceback" not in response.text
    assert any(
        isinstance(record.exc_info, tuple) and record.exc_info[0] is RuntimeError
        for record in caplog.records
    ), "the handler must log the original traceback"


def test_user_facing_error_bubbles(monkeypatch):
    """A UserFacingError that escapes its local catch still reaches the
    user verbatim — HTTP 200 bubble, no exception out of the app."""
    monkeypatch.setattr("app.controllers.chat.interpret", _user_facing)

    with TestClient(app) as client:
        token = _page_token(client)
        response = client.post("/chat", data={"token": token, "message": "Ana, Bob"})

    assert response.status_code == 200
    assert "custom notice" in response.text


def test_lifespan_starts_sweep_task():
    with TestClient(app):
        task = app.state.sweep_task
        assert isinstance(task, asyncio.Task)
        assert not task.done()
    assert task.cancelled(), "sweep task must be cancelled on shutdown"


def test_sweep_collects_orphaned_session(monkeypatch):
    """The sweep loop really runs: an orphan past its TTL disappears."""
    monkeypatch.setattr(main, "SWEEP_INTERVAL_SECONDS", 0.01)
    session = app.state.store.create()
    session.created_at = time.time() - 7200  # default TTL is 3600 s

    with TestClient(app):
        deadline = time.monotonic() + 2.0
        while (
            app.state.store.get(session.token) is not None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert (
            app.state.store.get(session.token) is None
        ), "sweep task must collect sessions past their TTL"

    assert app.state.store.get(session.token) is None
