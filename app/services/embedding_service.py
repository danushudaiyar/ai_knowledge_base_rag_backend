# Chunks → embeddings
from typing import List
import numpy as np
from app.core.logging import logger
from app.core.exceptions import AppException


def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for text chunks (currently using dummy vectors)
    
    Args:
        chunks: List of text chunks
    
    Returns:
        List of embedding vectors (dummy vectors for now)
    
    Raises:
        AppException: If embedding generation fails
    """
    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        if not chunks:
            raise AppException("No chunks provided for embedding generation", status_code=400)
        
        embeddings = []
        for i, chunk in enumerate(chunks):
            # Generate dummy embedding vector (768 dimensions)
            dummy_vector = np.random.rand(768).tolist()
            embeddings.append(dummy_vector)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise AppException(f"Failed to generate embeddings: {str(e)}", status_code=500)
