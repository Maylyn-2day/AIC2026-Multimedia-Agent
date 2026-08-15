"""
Pydantic Schemas Package.

All request/response models for the 7 RESTful API endpoints.
"""

from backend.schemas.common import BaseResponse, ErrorDetail

__all__ = ["BaseResponse", "ErrorDetail"]
