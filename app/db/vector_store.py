# Chroma/FAISS setup
from typing import List
import chromadb
from chromadb.config import Settings
from app.core.logging import logger
import uuid


# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)


def store_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """
    Store text chunks and their embeddings in ChromaDB
    
    Args:
        chunks: List of text chunks
        embeddings: List of embedding vectors
    """
    logger.info(f"Storing {len(chunks)} chunks in vector store")
    
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)})")
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    
    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    
    logger.info(f"Successfully stored {len(chunks)} chunks in vector store")
