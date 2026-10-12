"""Image mime-type sniffing from magic bytes.

Shared by the Gemini client (labels vision uploads on the wire) and the
media route (labels stored session images when serving them) — one
definition of what an image *is*, defaulting to ``image/png`` for bytes
it cannot recognize.
"""

# The three upload formats the app accepts (see the chat controller's
# _ALLOWED_UPLOAD_TYPES); unknown bytes default to PNG.
PNG_MIME = "image/png"
JPEG_MIME = "image/jpeg"
WEBP_MIME = "image/webp"
DEFAULT_MIME = PNG_MIME


def detect_mime(image: bytes) -> str:
    """Sniff the image's mime from magic bytes; default to ``image/png``.

    The app accepts PNG, JPEG and WEBP uploads; labeling one as another
    makes consumers reject the image (Gemini, strict proxies), so the
    type travels with the bytes.
    """
    if image.startswith(b"\x89PNG"):
        return PNG_MIME
    if image.startswith(b"\xff\xd8\xff"):
        return JPEG_MIME
    if len(image) >= 12 and image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return WEBP_MIME
    return DEFAULT_MIME
