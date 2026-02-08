"""API routers."""

from app.routers.search import router as search_router
from app.routers.feedback import router as feedback_router

__all__ = ["search_router", "feedback_router"]
