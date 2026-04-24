# File → text → chunks
from fastapi import UploadFile
from typing import List
from app.utils.file_parser import parse_file
from app.utils.chunker import chunk_text
from app.core.logging import logger


async def process_file(file: UploadFile) -> List[str]:
    """
    Process uploaded file: parse and chunk the content
    
    Args:
        file: UploadFile object from FastAPI
    
    Returns:
        List of text chunks
    """
    logger.info(f"Starting file processing: {file.filename}")
    
    # Parse file to extract text
    text = await parse_file(file)
    
    # Chunk the text
    chunks = chunk_text(text)
    
    logger.info(f"File processing complete: {file.filename}, {len(chunks)} chunks created")
    return chunks
