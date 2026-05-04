# File → text → chunks
from fastapi import UploadFile
from typing import List
from app.utils.file_parser import parse_file, parse_url
from app.utils.chunker import chunk_text
from app.services.embedding_service import generate_embeddings
from app.db.vector_store import store_embeddings
from app.core.logging import logger
from app.core.exceptions import AppException
from app.config import settings


async def process_file(file: UploadFile) -> dict:
    """
    Process uploaded file: parse, chunk, embed, and store
    
    Args:
        file: UploadFile object from FastAPI
    
    Returns:
        Dictionary with processing results
    
    Raises:
        AppException: If file processing fails
    """
    try:
        logger.info(f"Starting file processing: {file.filename}")
        
        # Parse file to extract text
        text = await parse_file(file)
        
        if not text or not text.strip():
            raise AppException(f"No text content extracted from file: {file.filename}", status_code=400)
        
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
        
        if not chunks:
            raise AppException(f"No chunks created from file: {file.filename}", status_code=400)
        
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
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise AppException(f"Failed to process file: {str(e)}", status_code=500)


async def process_url(url: str) -> dict:
    """
    Process URL: fetch HTML, parse, chunk, embed, and store
    
    Args:
        url: URL to fetch and process
    
    Returns:
        Dictionary with processing results
    
    Raises:
        AppException: If URL processing fails
    """
    try:
        logger.info(f"Starting URL processing: {url}")
        
        # Fetch and parse URL to extract text
        text = await parse_url(url)
        
        if not text or not text.strip():
            raise AppException(f"No text content extracted from URL: {url}", status_code=400)
        
        # Chunk the text with metadata using configured parameters
        chunk_data = chunk_text(
            text, 
            filename=url,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        
        # Extract text chunks and metadata
        chunks = [item["text"] for item in chunk_data]
        metadatas = [item["metadata"] for item in chunk_data]
        
        if not chunks:
            raise AppException(f"No chunks created from URL: {url}", status_code=400)
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Store in vector database with metadata
        store_embeddings(chunks, embeddings, metadatas)
        
        logger.info(f"URL processing complete: {url}, {len(chunks)} chunks created and stored")
        
        return {
            "url": url,
            "chunks_count": len(chunks),
            "status": "success"
        }
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise AppException(f"Failed to process URL: {str(e)}", status_code=500)
