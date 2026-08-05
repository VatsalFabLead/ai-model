#!/usr/bin/env python3
"""Passenger / Hostinger shared-hosting entrypoint (bridges FastAPI ASGI -> WSGI & CGI)."""

import os
import sys
from pathlib import Path

# 1. Add workspace root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2. Automatically activate .venv site-packages if present
VENV_LIB = ROOT / ".venv" / "lib"
if VENV_LIB.exists():
    for site_pkg in VENV_LIB.glob("python*/site-packages"):
        if str(site_pkg) not in sys.path:
            sys.path.insert(0, str(site_pkg))

# 3. Create WSGI application
try:
    from a2wsgi import ASGIMiddleware
    from app.main import app
    application = ASGIMiddleware(app)
except Exception as err:
    import traceback
    error_tb = traceback.format_exc()
    def application(environ, start_response):
        status = '500 Internal Server Error'
        response_headers = [('Content-type', 'text/plain; charset=utf-8')]
        start_response(status, response_headers)
        return [f"Startup Error:\n{err}\n\n{error_tb}".encode('utf-8')]

# 4. Support direct CGI execution if Phusion Passenger is not handling .py directly
if __name__ == "__main__" or "GATEWAY_INTERFACE" in os.environ:
    from wsgiref.handlers import CGIHandler
    CGIHandler().run(application)
