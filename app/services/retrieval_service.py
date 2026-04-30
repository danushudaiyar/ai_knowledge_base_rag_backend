# Query → relevant chunks
from typing import List, Dict, Any
from app.services.embedding_service import generate_embeddings
from app.db.vector_store import collection
from app.core.logging import logger
from app.config import settings


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from the vector database based on a query
    
    Args:
        query: The user's question/query
        top_k: Number of top results to return (uses config default if None)
    
    Returns:
        List of dictionaries containing retrieved chunks and metadata
    """
    # Use config default if not specified
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K
    
    # Enforce maximum limit
    top_k = min(top_k, settings.RETRIEVAL_MAX_TOP_K)
    
    logger.info(f"Retrieving top {top_k} chunks for query: {query}")
    
    # Convert query to embedding
    query_embedding = generate_embeddings([query])[0]
    
    # Query vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    chunks = []
    if results['documents'] and results['documents'][0]:
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
    
    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks
