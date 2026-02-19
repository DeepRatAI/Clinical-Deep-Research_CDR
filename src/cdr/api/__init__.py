"""
CDR API Layer

FastAPI interface.
"""

from cdr.api.routes import router, create_app

__all__ = ["create_app", "router"]
