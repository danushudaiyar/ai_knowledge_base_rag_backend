# Query routes for RAG-based Q&A
from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.core.logging import logger
from app.services.retrieval_service import retrieve

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base with a question
    
    Args:
        request: QueryRequest containing the question
    
    Returns:
        QueryResponse with answer and sources
    """
    logger.info(f"Query received: {request.question}")
    
    # Retrieve relevant chunks from vector database
    chunks = retrieve(request.question, top_k=3)
    
    # Extract chunk contents for response
    chunk_contents = [chunk['content'] for chunk in chunks]
    
    # Return retrieved chunks (answer generation will be added later)
    return QueryResponse(
        answer=f"Retrieved {len(chunks)} relevant chunks:\n\n" + "\n\n".join(chunk_contents) if chunks else "No relevant information found.",
        sources=[f"Chunk {i+1}" for i in range(len(chunks))]
    )
