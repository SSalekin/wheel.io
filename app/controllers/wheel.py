from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    # A fresh session per page load: refresh ⇒ new token ⇒ fresh start.
    session = request.app.state.store.create()
    return request.app.state.templates.TemplateResponse(
        request, "base.html", {"session": session}
    )


@router.get("/healthz")
def healthz():
    return {"status": "ok"}
