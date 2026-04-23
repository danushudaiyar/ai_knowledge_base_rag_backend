# Upload routes for document ingestion
from fastapi import APIRouter, UploadFile, File
from app.models.schemas import UploadResponse
from app.core.logging import logger

router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for ingestion into the knowledge base
    
    Args:
        file: Uploaded document file (PDF, TXT, etc.)
    
    Returns:
        UploadResponse with status message
    """
    logger.info(f"File received: {file.filename}")
    
    return UploadResponse(message="file received")
