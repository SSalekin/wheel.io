"""Media controller: serve in-memory session images (face crops).

Photos live only in ``session.media`` keyed by the per-page token — no
persistence of any kind. A token from another page, an expired session,
or an unknown id all degrade to a plain 404, which the wheel renders as
a label-only segment.
"""

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


@router.get("/media/{token}/{media_id}")
def media(token: str, media_id: str, request: Request) -> Response:
    """Serve one stored image as PNG; unknown token or id ⇒ 404."""
    session = request.app.state.store.get(token)
    if session is None or media_id not in session.media:
        return Response(status_code=404)
    return Response(content=session.media[media_id], media_type="image/png")
