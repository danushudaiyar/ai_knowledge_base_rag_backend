# Upload routes for document ingestion
from fastapi import APIRouter, UploadFile, File
from app.models.schemas import UploadResponse, URLUploadRequest
from app.core.logging import logger
from app.services.ingestion_service import process_file, process_url

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
    
    # Process file: parse and chunk
    chunks = await process_file(file)
    
    # Log chunk count
    logger.info(f"File processed successfully: {len(chunks)} chunks created")
    
    return UploadResponse(message=f"file received and processed into {len(chunks)} chunks")


@router.post("/upload-url", response_model=UploadResponse)
async def upload_url(request: URLUploadRequest):
    """
    Upload a URL for ingestion into the knowledge base
    Fetches HTML content from the URL and extracts text
    
    Args:
        request: URLUploadRequest containing the URL to fetch
    
    Returns:
        UploadResponse with status message
    """
    logger.info(f"URL received: {request.url}")
    
    # Process URL: fetch, parse, and chunk
    result = await process_url(request.url)
    
    # Log chunk count
    logger.info(f"URL processed successfully: {result['chunks_count']} chunks created")
    
    return UploadResponse(message=f"URL fetched and processed into {result['chunks_count']} chunks")
