# Wheelio

A chat-driven random-name wheel, in the spirit of [wheelofnames.com](https://wheelofnames.com/) — but you never build the wheel by hand. Type a list, drop in a screenshot, or attach student photos: the chat answers with a spinning wheel instead of prose.

```
You:  wheel: Ana, Bob, Chi, Dani
      ┌─────────────── chat ───────────────┐   ┌──────── wheel ────────┐
You:  │ wheel: Ana, Bob, Chi, Dani      ▸  │   │      ╭─ canvas ─╮     │
Sys:  │ 🎉 wheel ready (4 entries)      ◂  │   │      │ segments │     │
      └────────────────────────────────────┘   │      ╰──────────╯     │
                                               │     [ Go / Start ]    │
                                               │  ☐ Remove picked      │
                                               └───────────────────────┘
```

## What it does

- **Type a list** → `wheel: Ana, Bob, Chi` (or a bulleted/numbered/multi-line list) builds the wheel.
- **Attach one image** → a screenshot of a roster is read locally with OCR, layout is reconstructed into a table, and the column you asked for ("switch to id", "pick the names…") becomes the entries. A photo of people gets its faces detected, cropped, and turned into `Face 1…N` entries with thumbnails.
- **Attach multiple images** → each file becomes an entry, labeled by its filename (`student_a.png` → `Student A`).
- **Follow up in chat:**
  | You type | It does |
  |---|---|
  | `add Sam, Tia` | appends entries |
  | `remove Bob` | removes an entry (case-insensitive; warns if duplicates remain) |
  | `switch to id` | rebuilds the wheel from another column of the last uploaded table |
  | `reset` / `clear` | empties the wheel, picks, and uploads |
  | `spin` / `go` | same as pressing the Go button |
- **Spin the wheel** — the server picks the winner (uniform, `SystemRandom`); the browser only animates. Check **Remove the picked students from wheel** (default off) to move winners to the **Picked** sidebar as you go; the `🎉 {label}` banner records every win either way.

## Requirements

- Python 3.11+
- System **Tesseract ≥ 5** with English data:
  - Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr tesseract-ocr-eng`
  - macOS: `brew install tesseract`
- A **Gemini API key** — *optional*. Without it (or on timeout/error) Wheelio degrades gracefully: local rules handle simple messages silently, and when the AI was needed the reply carries a `⚠️ AI unreachable — best-guess parsing.` note.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env      # then uncomment GEMINI_API_KEY and paste your key
```

## Run

```bash
uvicorn app.main:app
```

Open http://127.0.0.1:8000 and start chatting.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | Enables AI-assisted parsing (ambiguous messages, low-confidence OCR, vision fallback) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
| `GEMINI_TIMEOUT_S` | `10` | Per-request timeout before degrading to rules |

## How it works

Every message becomes one of nine intents (`create_text`, `create_image`, `create_images`, `add`, `remove`, `switch_column`, `reset`, `spin`, `clarify`):

1. **Local rules first** — pattern matching handles typed lists and all follow-ups with zero network calls.
2. **Gemini Flash for the tricky cases** — unmatched messages go to the model with a compact state summary; it returns the same validated intent shape, so there's one downstream code path either way.
3. **Always degrade, never crash** — model timeout, bad JSON, or missing key falls back to the strongest rule-like guess plus the ⚠️ note; unexpected errors become a friendly `Something went wrong.` bubble with the traceback in the server log.

Images: uploads (png/jpeg/webp ≤ 10 MB each, up to 20 files) run through local **Tesseract OCR** → line/column layout reconstruction → column selection (header keywords → Gemini if ambiguous). If the reading itself is unreliable (mean confidence < 60), the image escalates to Gemini vision. Face photos go through **OpenCV YuNet** (`face_detection_yunet.onnx`, bundled in `app/models/` — zero setup).

State lives in memory, keyed by a per-page token: **refreshing the page starts over** — no database, no localStorage.

## Tests

```bash
pytest -v
```

171 tests: intent rules, OCR integration against real Tesseract, face-detection wiring, column selection, spin/state semantics, and end-to-end chat flows (the Gemini HTTP boundary is mocked with `respx`).

## Project structure

```
app/
├── main.py           # app factory, exception handlers, session TTL sweep
├── controllers/      # chat, spin, toggle, wheel, media routes
├── services/         # rules, intent_router, gemini, ocr, column_picker,
│                     # faces, images, mime, wheel_ops, errors
├── models/           # Entry, WheelSession, Intent, Table (pydantic)
│                     # + face_detection_yunet.onnx (YuNet model asset)
├── state/            # InMemoryStateStore (TTL-expired sessions)
├── views/            # base.html + HTMX partials
└── static/           # htmx.min.js (vendored), wheel.js, styles.css
tests/
```

Server-rendered HTML + HTMX partials, one hand-written CSS file, vanilla JS for the canvas — no frameworks, no build step.
