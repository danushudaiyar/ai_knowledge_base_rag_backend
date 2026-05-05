# Query → relevant chunks
from typing import List, Dict, Any
from app.services.embedding_service import generate_embeddings
from app.db.vector_store import collection
from app.core.logging import logger
from app.core.exceptions import AppException
from app.config import settings


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from the vector database based on a query
    
    Args:
        query: The user's question/query
        top_k: Number of top results to return (uses config default if None)
    
    Returns:
        List of dictionaries containing retrieved chunks and metadata
    
    Raises:
        AppException: If retrieval fails
    """
    try:
        if not query or not query.strip():
            raise AppException("Query cannot be empty", status_code=400)
        
        # Use config default if not specified
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K
        
        # Enforce maximum limit
        top_k = min(top_k, settings.RETRIEVAL_MAX_TOP_K)
        
        if top_k <= 0:
            raise AppException("top_k must be greater than 0", status_code=400)
        
        logger.info(f"Retrieving top {top_k} chunks for query: '{query[:100]}...'" if len(query) > 100 else f"Retrieving top {top_k} chunks for query: '{query}'")
        
        # Convert query to embedding
        logger.debug(f"Generating embedding for query")
        query_embedding = generate_embeddings([query])[0]
        logger.debug(f"Query embedding generated, dimension: {len(query_embedding)}")
        
        # Query vector database
        logger.debug(f"Querying vector database with n_results={top_k}")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        chunks = []
        if results['documents'] and results['documents'][0]:
            logger.debug(f"Processing {len(results['documents'][0])} results from vector database")
            for i, doc in enumerate(results['documents'][0]):
                chunk_data = {
                    'content': doc,
                    'id': results['ids'][0][i] if results['ids'] else None,
                    'distance': results['distances'][0][i] if results.get('distances') else None
                }
                # Add metadata if available
                if results.get('metadatas') and results['metadatas'][0]:
                    chunk_data['metadata'] = results['metadatas'][0][i]
                
                chunks.append(chunk_data)
        
        logger.info(f"Successfully retrieved {len(chunks)} chunks for query")
        if chunks and chunks[0].get('distance') is not None:
            logger.debug(f"Best match distance: {chunks[0]['distance']:.4f}")
        return chunks
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chunks for query '{query}': {str(e)}")
        raise AppException(f"Failed to retrieve chunks: {str(e)}", status_code=500)
