# FastAPI entry point
from fastapi import FastAPI
from app.routes import upload_routes, query_routes, health_routes
from app.core.exceptions import AppException, app_exception_handler
from app.core.logging import logger

app = FastAPI(
    title="AI Knowledge Base & Q&A API",
    description="RAG-based document ingestion and question answering system",
    version="1.0.0"
)

# Register exception handler
app.add_exception_handler(AppException, app_exception_handler)

# Register routers
app.include_router(upload_routes.router, prefix="/api/v1")
app.include_router(query_routes.router, prefix="/api/v1")
app.include_router(health_routes.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    logger.info("App started")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("App shutdown")
