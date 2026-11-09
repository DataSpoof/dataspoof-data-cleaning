"""ASGI entry point: ``uvicorn automatedcleaning.deploy.asgi:app`` (same environment variables
as :mod:`automatedcleaning.deploy.wsgi`)."""

import logging
import os

from .server import make_asgi_app
from .wsgi import build_app

logging.basicConfig(level=os.environ.get("AC_LOG_LEVEL", "INFO"))
app = make_asgi_app(wsgi_app=build_app())
