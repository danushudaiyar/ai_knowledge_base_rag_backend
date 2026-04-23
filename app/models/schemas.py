# Request/response schemas
from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """
    Request model for question answering
    """
    question: str = Field(..., description="Question to ask the knowledge base", min_length=1)


class QueryResponse(BaseModel):
    """
    Response model for question answering
    """
    answer: str = Field(..., description="Generated answer based on retrieved documents")
    sources: List[str] = Field(default_factory=list, description="Source documents used for the answer")


class UploadResponse(BaseModel):
    """
    Response model for file upload
    """
    message: str = Field(..., description="Upload status message")
