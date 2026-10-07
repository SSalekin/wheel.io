from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.controllers import wheel

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

app = FastAPI(title="Wheelio")

# Shared template instance; controllers render through app.state.templates.
app.state.templates = Jinja2Templates(directory=BASE_DIR / "views")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

app.include_router(wheel.router)
