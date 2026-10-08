import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.controllers import bubble, chat, media, spin, toggle, wheel
from app.services.errors import UserFacingError
from app.state.store import InMemoryStateStore

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

logger = logging.getLogger(__name__)

# Pinned copy: the generic bubble when anything unexpected escapes a route.
GENERIC_ERROR = "Something went wrong."

# Lifespan sweep cadence: collect orphaned sessions this often.
SWEEP_INTERVAL_SECONDS = 60


async def _sweep_loop(store: InMemoryStateStore) -> None:
    """Run ``store.expire()`` every ``SWEEP_INTERVAL_SECONDS`` until cancelled."""
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        store.expire()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sweep_task = asyncio.create_task(_sweep_loop(app.state.store))
    try:
        yield
    finally:
        app.state.sweep_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.sweep_task


app = FastAPI(title="Wheelio", lifespan=lifespan)

# Shared template instance; controllers render through app.state.templates.
app.state.templates = Jinja2Templates(directory=BASE_DIR / "views")

# In-memory session store; a refresh mints a new token, orphaning the old one.
app.state.store = InMemoryStateStore()


@app.exception_handler(UserFacingError)
async def user_facing_error_handler(request: Request, exc: UserFacingError) -> HTMLResponse:
    """Safety net for UserFacingError escaping a route: HTTP 200 bubble."""
    return HTMLResponse(bubble(request, "system", exc.user_message))


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> HTMLResponse:
    """Last resort: HTTP 200 generic bubble; the traceback goes to the logs
    here and again to the server when Starlette re-raises."""
    logger.exception("Unhandled exception while serving %s", request.url.path)
    return HTMLResponse(bubble(request, "system", GENERIC_ERROR))


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(wheel.router)
app.include_router(chat.router)
app.include_router(media.router)
app.include_router(spin.router)
app.include_router(toggle.router)
