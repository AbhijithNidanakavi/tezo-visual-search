from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.schemas import HealthResponse, SearchResponse
from app.search_engine import SearchEngine


engine = SearchEngine(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    if engine.store.exists() and not engine.is_ready:
        engine.load()
    yield


app = FastAPI(
    title="Enterprise Visual Search",
    description="Semantic image retrieval with AI-guided explanations.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
app.mount("/data", StaticFiles(directory=str(settings.data_dir)), name="data")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def sync_engine_state() -> None:
    engine.refresh_if_stale()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    sync_engine_state()
    return HealthResponse(
        status="ok",
        index_ready=engine.is_ready,
        indexed_images=len(engine.records),
        embedder_backend=engine._embedder.name if engine._embedder is not None else engine.settings.embedder_backend,
        index_image_count=engine.config.get("image_count") if engine.config else None,
    )


@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., min_length=1), top_k: int = Query(default=settings.default_top_k, ge=1, le=20)) -> SearchResponse:
    sync_engine_state()
    try:
        return engine.search(query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home(request: Request, query: str | None = None) -> HTMLResponse:
    sync_engine_state()
    response = None
    error = None
    if query:
        try:
            response = engine.search(query)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "query": query or "",
            "response": response,
            "error": error,
            "index_ready": engine.is_ready,
            "indexed_images": len(engine.records),
            "index_image_count": engine.config.get("image_count") if engine.config else None,
            "embedder_backend": engine.config.get("embedder_backend", engine.settings.embedder_backend),
        },
    )
