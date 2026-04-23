# Query routes for RAG-based Q&A
from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.core.logging import logger

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
    
    # Dummy response for now
    return QueryResponse(
        answer="This is a dummy response. RAG pipeline will be implemented soon.",
        sources=["dummy_source_1.pdf", "dummy_source_2.pdf"]
    )
