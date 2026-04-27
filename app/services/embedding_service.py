# Chunks → embeddings
from typing import List
import numpy as np
from app.core.logging import logger


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for text chunks (currently using dummy vectors)
    
    Args:
        chunks: List of text chunks
    
    Returns:
        List of embedding vectors (dummy vectors for now)
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        # Generate dummy embedding vector (768 dimensions)
        dummy_vector = np.random.rand(768).tolist()
        embeddings.append(dummy_vector)
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings
