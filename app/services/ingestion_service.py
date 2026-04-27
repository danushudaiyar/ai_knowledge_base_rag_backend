# File → text → chunks
from fastapi import UploadFile
from typing import List
from app.utils.file_parser import parse_file
from app.utils.chunker import chunk_text
from app.services.embedding_service import generate_embeddings
from app.db.vector_store import store_embeddings
from app.core.logging import logger


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
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    
    # Store in vector database
    store_embeddings(chunks, embeddings)
    
    logger.info(f"File processing complete: {file.filename}, {len(chunks)} chunks created and stored")
    
    return {
        "filename": file.filename,
        "chunks_count": len(chunks),
        "status": "success"
    }
