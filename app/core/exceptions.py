# Custom exceptions and handlers
from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """
    Custom application exception
    """
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


async def app_exception_handler(request: Request, exc: AppException):
    """
    Global exception handler for AppException
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message}
    )
