# Query → relevant chunks
from typing import List, Dict, Any
from app.services.embedding_service import generate_embeddings
from app.db.vector_store import collection
from app.core.logging import logger


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from the vector database based on a query
    
    Args:
        query: The user's question/query
        top_k: Number of top results to return (default: 3)
    
    Returns:
        List of dictionaries containing retrieved chunks and metadata
    """
    logger.info(f"Retrieving chunks for query: {query}")
    
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
            chunks.append({
                'content': doc,
                'id': results['ids'][0][i] if results['ids'] else None,
                'distance': results['distances'][0][i] if results.get('distances') else None
            })
    
    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks
