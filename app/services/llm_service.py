# Context → answer generation
import os
from app.core.logging import logger
from app.core.exceptions import AppException


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer based on the query and retrieved context.
    
    Args:
        query: The user's question
        context: Retrieved context from the vector database
    
    Returns:
        Generated answer (currently a dummy response)
    
    Raises:
        AppException: If answer generation fails
    """
    try:
        # Read the RAG prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "rag_prompt.txt")
        
        if not query or not query.strip():
            raise AppException("Query cannot be empty", status_code=400)
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found at {prompt_path}")
            raise AppException("Prompt template not found", status_code=500)
        
        # Inject query and context into the template
        final_prompt = prompt_template.format(query=query, context=context)
        
        logger.info(f"Generated prompt for LLM (length: {len(final_prompt)} chars)")
        
        # Return a dummy LLM response for now
        # TODO: Replace with actual LLM API call (OpenAI, Anthropic, etc.)
        dummy_response = f"Based on the provided context, here's what I found regarding your question: '{query}'. This is a dummy response that will be replaced with actual LLM integration."
        
        return dummy_response
        
    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise AppException(f"Failed to generate answer: {str(e)}", status_code=500)
