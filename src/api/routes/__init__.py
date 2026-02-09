"""API routes module."""

from src.api.routes.enrollment import router as enrollment_router
from src.api.routes.identification import router as identification_router
from src.api.routes.speakers import router as speakers_router
from src.api.routes.admin import router as admin_router

__all__ = [
    "enrollment_router",
    "identification_router",
    "speakers_router",
    "admin_router",
]
