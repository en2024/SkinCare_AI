"""ASGI entry point — used for async deployments (e.g., Daphne, Uvicorn)."""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
application = get_asgi_application()
