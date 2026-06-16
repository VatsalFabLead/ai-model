"""WSGI/ASGI entry for gunicorn on Hostinger VPS."""

from app.main import app

__all__ = ["app"]
