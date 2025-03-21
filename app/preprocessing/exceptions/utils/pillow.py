"""
Exceptions connected with image and Pillow library are defined here.
"""

from fastapi import status

from app.risk_calculation.exceptions import IduApiError


class InvalidImageError(IduApiError):
    """
    Exception to raise when user upload invalid imagee.
    """

    def __init__(self, project_id: int):
        """
        Construct from requested identifier and entity (table) name.
        """
        self.project_id = project_id
        super().__init__()

    def __str__(self) -> str:
        return f"Invalid image for project with id = {self.project_id} was uploaded."

    def get_status_code(self) -> int:
        """
        Return '400 Bad Request' status code.
        """
        return status.HTTP_400_BAD_REQUEST
