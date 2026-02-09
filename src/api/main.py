"""
FastAPI application for Speaker Identification System.

Main entry point for the REST API.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from config.logging_config import setup_logging, get_logger
from src.api.dependencies import initialize_services, shutdown_services
from src.api.routes import (
    enrollment_router,
    identification_router,
    speakers_router,
    admin_router,
)
from src.api.schemas import HealthResponse

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("application_starting")
    await initialize_services()
    logger.info("application_started")

    yield

    # Shutdown
    logger.info("application_stopping")
    await shutdown_services()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        ## Speaker Identification System

        Production-ready speaker identification using NVIDIA NeMo embeddings.

        ### Features
        - **Enrollment**: Register speakers with audio samples
        - **Identification**: Identify speakers from audio
        - **Management**: List, search, and remove speakers
        - **Configuration**: Adjust thresholds and settings

        ### Quick Start
        1. Enroll speakers using POST /api/v1/enroll
        2. Identify speakers using POST /api/v1/identify
        3. Manage speakers using GET/DELETE /api/v1/speakers
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(enrollment_router, prefix=settings.api_prefix)
    app.include_router(identification_router, prefix=settings.api_prefix)
    app.include_router(speakers_router, prefix=settings.api_prefix)
    app.include_router(admin_router, prefix=settings.api_prefix)

    # Mount static files
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
        summary="Health check",
    )
    async def health_check() -> HealthResponse:
        """Check if the service is healthy."""
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            timestamp=datetime.utcnow(),
        )

    # Root endpoint - serve UI
    @app.get("/", tags=["root"], include_in_schema=False)
    async def root():
        """Serve the web UI."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "api_prefix": settings.api_prefix,
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
            },
        )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers,
    )
