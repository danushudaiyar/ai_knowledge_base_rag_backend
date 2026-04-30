# Text splitting logic
from typing import List, Dict, Any
from app.core.logging import logger


def chunk_text(text: str, filename: str = "unknown", chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split text into chunks of specified size with overlap and metadata
    
    Args:
        text: Input text to split
        filename: Name of the source file
        chunk_size: Maximum size of each chunk (default: 500)
        overlap: Number of characters to overlap between chunks (default: 50)
    
    Returns:
        List of dictionaries containing text chunks and metadata
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for chunking")
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    chunk_id = 0
    
    while start < text_length:
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        
        # Add chunk if it's not empty
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_id": chunk_id
                }
            })
            chunk_id += 1
        
        # Move start position, accounting for overlap
        start = end - overlap if end < text_length else text_length
    
    logger.info(f"Text split into {len(chunks)} chunks for file: {filename}")
    return chunks
