"""Passenger/Hostinger shared-hosting entry (bridges FastAPI ASGI → WSGI).

Upload this file to public_html and point hPanel Python App startup here.
No GPT/Claude/Gemini — same custom stack as wsgi.py / gunicorn.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from a2wsgi import ASGIMiddleware

from app.main import app

application = ASGIMiddleware(app)
