# Query routes for RAG-based Q&A
from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.core.logging import logger
from app.services.retrieval_service import retrieve
from app.services.llm_service import generate_answer

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
    
    if not chunks:
        return QueryResponse(
            answer="No relevant information found.",
            sources=[]
        )
    
    # Prepare context from retrieved chunks
    context = "\n\n".join([chunk['content'] for chunk in chunks])
    
    # Generate answer using LLM
    answer = generate_answer(request.question, context)
    
    # Extract sources from chunks (metadata like filenames if available)
    sources = [chunk.get('metadata', {}).get('source', f"Chunk {i+1}") 
               for i, chunk in enumerate(chunks)]
    
    return QueryResponse(
        answer=answer,
        sources=sources
    )
