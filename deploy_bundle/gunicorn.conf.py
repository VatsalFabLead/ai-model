# Hostinger / production process config (free stack: gunicorn + uvicorn workers)
import multiprocessing
import os

bind = os.getenv("BIND", "0.0.0.0:8000")
workers = int(os.getenv("WEB_CONCURRENCY", max(2, multiprocessing.cpu_count())))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
