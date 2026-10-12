import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.state.store import InMemoryStateStore


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """No test may reach the real Gemini API, and every test starts from
    a fresh store — sessions never leak across tests."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    original = app.state.store
    app.state.store = InMemoryStateStore()
    yield
    app.state.store = original


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
