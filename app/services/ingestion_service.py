# File → text → chunks
from fastapi import UploadFile
from typing import List
from app.utils.file_parser import parse_file
from app.utils.chunker import chunk_text
from app.services.embedding_service import generate_embeddings
from app.db.vector_store import store_embeddings
from app.core.logging import logger
from app.config import settings


async def process_file(file: UploadFile) -> dict:
    """
    Process uploaded file: parse, chunk, embed, and store
    
    Args:
        file: UploadFile object from FastAPI
    
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting file processing: {file.filename}")
    
    # Parse file to extract text
    text = await parse_file(file)
    
    # Chunk the text with metadata using configured parameters
    chunk_data = chunk_text(
        text, 
        filename=file.filename,
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP
    )
    
    # Extract text chunks and metadata
    chunks = [item["text"] for item in chunk_data]
    metadatas = [item["metadata"] for item in chunk_data]
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    
    # Store in vector database with metadata
    store_embeddings(chunks, embeddings, metadatas)
    
    logger.info(f"File processing complete: {file.filename}, {len(chunks)} chunks created and stored")
    
    return {
        "filename": file.filename,
        "chunks_count": len(chunks),
        "status": "success"
    }
