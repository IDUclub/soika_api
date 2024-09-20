"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""

from fastapi import APIRouter
from loguru import logger

router = APIRouter()


@router.get("/")
async def get_content():
    logger.info("Get content")
    return "Get content"
