# Text splitting logic
from typing import List
from app.core.logging import logger


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of specified size with overlap
    
    Args:
        text: Input text to split
        chunk_size: Maximum size of each chunk (default: 500)
        overlap: Number of characters to overlap between chunks (default: 50)
    
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for chunking")
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        
        # Add chunk if it's not empty
        if chunk.strip():
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap if end < text_length else text_length
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks
