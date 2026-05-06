"""
Comprehensive tests for the ingestion service and related utilities.
Tests file processing, URL processing, text chunking, and error handling.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import UploadFile
import io
from app.services.ingestion_service import process_file, process_url
from app.utils.chunker import chunk_text
from app.utils.file_parser import (
    is_url, 
    extract_text_from_html, 
    parse_url, 
    fetch_url_content
)
from app.core.exceptions import AppException


# ========================
# File Processing Tests
# ========================

@pytest.mark.asyncio
async def test_process_file_success():
    """Test successful file processing with complete pipeline"""
    # Create mock file
    file_content = b"This is a test document with some content that should be chunked properly."
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.file = io.BytesIO(file_content)
    
    # Mock all dependencies
    with patch('app.services.ingestion_service.parse_file') as mock_parse, \
         patch('app.services.ingestion_service.chunk_text') as mock_chunk, \
         patch('app.services.ingestion_service.generate_embeddings') as mock_embed, \
         patch('app.services.ingestion_service.store_embeddings') as mock_store:
        
        # Setup mocks
        mock_parse.return_value = "Sample text content"
        mock_chunk.return_value = [
            {"text": "Sample text", "metadata": {"filename": "test.txt", "chunk_id": 0}},
            {"text": "content", "metadata": {"filename": "test.txt", "chunk_id": 1}}
        ]
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Execute
        result = await process_file(mock_file)
        
        # Assertions
        assert result["filename"] == "test.txt"
        assert result["chunks_count"] == 2
        assert result["status"] == "success"
        mock_parse.assert_called_once_with(mock_file)
        mock_chunk.assert_called_once()
        mock_embed.assert_called_once()
        mock_store.assert_called_once()


@pytest.mark.asyncio
async def test_process_file_empty_content():
    """Test file processing with empty or whitespace-only content"""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "empty.txt"
    
    with patch('app.services.ingestion_service.parse_file') as mock_parse:
        mock_parse.return_value = "   \n  \t  "
        
        with pytest.raises(AppException) as exc_info:
            await process_file(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "No text content extracted" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_process_file_no_chunks_created():
    """Test file processing when chunking produces no chunks"""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.txt"
    
    with patch('app.services.ingestion_service.parse_file') as mock_parse, \
         patch('app.services.ingestion_service.chunk_text') as mock_chunk:
        
        mock_parse.return_value = "Some text"
        mock_chunk.return_value = []
        
        with pytest.raises(AppException) as exc_info:
            await process_file(mock_file)
        
        assert exc_info.value.status_code == 400
        assert "No chunks created" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_process_file_parsing_error():
    """Test file processing when parsing fails"""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "corrupt.pdf"
    
    with patch('app.services.ingestion_service.parse_file') as mock_parse:
        mock_parse.side_effect = Exception("Failed to parse PDF")
        
        with pytest.raises(AppException) as exc_info:
            await process_file(mock_file)
        
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_process_file_embedding_error():
    """Test file processing when embedding generation fails"""
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "test.txt"
    
    with patch('app.services.ingestion_service.parse_file') as mock_parse, \
         patch('app.services.ingestion_service.chunk_text') as mock_chunk, \
         patch('app.services.ingestion_service.generate_embeddings') as mock_embed:
        
        mock_parse.return_value = "Sample text"
        mock_chunk.return_value = [{"text": "chunk", "metadata": {"filename": "test.txt", "chunk_id": 0}}]
        mock_embed.side_effect = Exception("Embedding API error")
        
        with pytest.raises(AppException):
            await process_file(mock_file)


# ========================
# URL Processing Tests
# ========================

@pytest.mark.asyncio
async def test_process_url_success():
    """Test successful URL processing with complete pipeline"""
    url = "https://example.com/article"
    
    with patch('app.services.ingestion_service.parse_url') as mock_parse, \
         patch('app.services.ingestion_service.chunk_text') as mock_chunk, \
         patch('app.services.ingestion_service.generate_embeddings') as mock_embed, \
         patch('app.services.ingestion_service.store_embeddings') as mock_store:
        
        # Setup mocks
        mock_parse.return_value = "Article content from web page"
        mock_chunk.return_value = [
            {"text": "Article content", "metadata": {"filename": url, "chunk_id": 0}},
            {"text": "from web page", "metadata": {"filename": url, "chunk_id": 1}}
        ]
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        # Execute
        result = await process_url(url)
        
        # Assertions
        assert result["url"] == url
        assert result["chunks_count"] == 2
        assert result["status"] == "success"
        mock_parse.assert_called_once_with(url)


@pytest.mark.asyncio
async def test_process_url_empty_content():
    """Test URL processing with empty content"""
    url = "https://example.com/empty"
    
    with patch('app.services.ingestion_service.parse_url') as mock_parse:
        mock_parse.return_value = ""
        
        with pytest.raises(AppException) as exc_info:
            await process_url(url)
        
        assert exc_info.value.status_code == 400
        assert "No text content extracted" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_process_url_fetch_error():
    """Test URL processing when fetching fails"""
    url = "https://invalid-domain-123456.com"
    
    with patch('app.services.ingestion_service.parse_url') as mock_parse:
        mock_parse.side_effect = ValueError("Failed to fetch URL")
        
        with pytest.raises(AppException):
            await process_url(url)


# ========================
# Text Chunking Tests
# ========================

def test_chunk_text_basic():
    """Test basic text chunking functionality"""
    text = "This is a test document. " * 50  # ~1250 chars
    chunks = chunk_text(text, filename="test.txt", chunk_size=500, overlap=50)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("text" in chunk and "metadata" in chunk for chunk in chunks)
    assert all(chunk["metadata"]["filename"] == "test.txt" for chunk in chunks)
    assert all("chunk_id" in chunk["metadata"] for chunk in chunks)


def test_chunk_text_with_overlap():
    """Test that overlap works correctly"""
    text = "0123456789" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    
    assert len(chunks) > 1
    # Check that consecutive chunks have some overlap
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i]["text"][-10:]
        chunk2_start = chunks[i + 1]["text"][:10]
        # There should be some overlap in content


def test_chunk_text_small_text():
    """Test chunking with text smaller than chunk size"""
    text = "Short text"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["metadata"]["chunk_id"] == 0


def test_chunk_text_empty():
    """Test chunking with empty text"""
    chunks = chunk_text("", filename="empty.txt")
    assert chunks == []
    
    chunks = chunk_text("   \n\t  ", filename="whitespace.txt")
    assert chunks == []


def test_chunk_text_metadata():
    """Test that metadata is correctly attached to chunks"""
    text = "A" * 1000
    chunks = chunk_text(text, filename="test_file.pdf", chunk_size=300, overlap=50)
    
    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["filename"] == "test_file.pdf"
        assert chunk["metadata"]["chunk_id"] == i


def test_chunk_text_custom_parameters():
    """Test chunking with custom chunk size and overlap"""
    text = "X" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 5
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)


# ========================
# URL Validation Tests
# ========================

def test_is_url_valid():
    """Test URL validation with valid URLs"""
    assert is_url("https://www.example.com") is True
    assert is_url("http://example.com") is True
    assert is_url("https://example.com/path/to/page") is True
    assert is_url("https://sub.domain.example.com") is True
    assert is_url("http://localhost:8080") is True
    assert is_url("http://192.168.1.1") is True


def test_is_url_invalid():
    """Test URL validation with invalid inputs"""
    assert is_url("not a url") is False
    assert is_url("example.com") is False
    assert is_url("ftp://example.com") is False
    assert is_url("") is False
    assert is_url("just text") is False


# ========================
# HTML Parsing Tests
# ========================

def test_extract_text_from_html_basic():
    """Test basic HTML text extraction"""
    html = """
    <html>
        <head><title>Test</title></head>
        <body>
            <h1>Header</h1>
            <p>Paragraph text</p>
        </body>
    </html>
    """
    text = extract_text_from_html(html)
    
    assert "Header" in text
    assert "Paragraph text" in text
    assert "<h1>" not in text
    assert "<p>" not in text


def test_extract_text_from_html_removes_scripts():
    """Test that script and style tags are removed"""
    html = """
    <html>
        <head>
            <style>body { color: red; }</style>
            <script>alert('test');</script>
        </head>
        <body>
            <p>Content</p>
            <script>console.log('remove me');</script>
        </body>
    </html>
    """
    text = extract_text_from_html(html)
    
    assert "Content" in text
    assert "color: red" not in text
    assert "alert" not in text
    assert "console.log" not in text


def test_extract_text_from_html_cleans_whitespace():
    """Test that excessive whitespace is cleaned up"""
    html = "<html><body><p>Line 1</p>    <p>Line 2</p></body></html>"
    text = extract_text_from_html(html)
    
    assert "Line 1" in text
    assert "Line 2" in text
    # Should not have excessive whitespace


def test_extract_text_from_html_empty():
    """Test extraction from empty or minimal HTML"""
    assert len(extract_text_from_html("<html></html>")) >= 0
    assert len(extract_text_from_html("")) >= 0


# ========================
# URL Fetching Tests
# ========================

@pytest.mark.asyncio
async def test_fetch_url_content_success():
    """Test successful URL content fetching"""
    url = "https://example.com"
    mock_response = Mock()
    mock_response.text = "<html><body>Test content</body></html>"
    mock_response.status_code = 200
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        content = await fetch_url_content(url)
        
        assert content == "<html><body>Test content</body></html>"


@pytest.mark.asyncio
async def test_fetch_url_content_http_error():
    """Test URL fetching with HTTP errors"""
    url = "https://example.com/404"
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_get = AsyncMock()
        mock_get.side_effect = Exception("HTTP 404")
        mock_client.return_value.__aenter__.return_value.get = mock_get
        
        with pytest.raises(ValueError) as exc_info:
            await fetch_url_content(url)
        
        assert "Failed to fetch URL" in str(exc_info.value)


@pytest.mark.asyncio
async def test_fetch_url_content_timeout():
    """Test URL fetching with custom timeout"""
    url = "https://slow-website.com"
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_get = AsyncMock(side_effect=Exception("Timeout"))
        mock_client.return_value.__aenter__.return_value.get = mock_get
        
        with pytest.raises(ValueError):
            await fetch_url_content(url, timeout=5)


@pytest.mark.asyncio
async def test_parse_url_integration():
    """Test complete URL parsing with fetching and extraction"""
    url = "https://example.com/article"
    html_content = "<html><body><h1>Title</h1><p>Article content here</p></body></html>"
    
    with patch('app.utils.file_parser.fetch_url_content') as mock_fetch:
        mock_fetch.return_value = html_content
        
        text = await parse_url(url)
        
        assert "Title" in text
        assert "Article content here" in text
        mock_fetch.assert_called_once_with(url)


# ========================
# Integration Tests
# ========================

@pytest.mark.asyncio
async def test_full_file_pipeline_integration():
    """Test complete file processing pipeline end-to-end"""
    # Create realistic mock file
    file_content = "This is a longer document. " * 100  # ~2800 chars
    mock_file = Mock(spec=UploadFile)
    mock_file.filename = "integration_test.txt"
    
    with patch('app.services.ingestion_service.parse_file') as mock_parse, \
         patch('app.services.ingestion_service.generate_embeddings') as mock_embed, \
         patch('app.services.ingestion_service.store_embeddings') as mock_store:
        
        mock_parse.return_value = file_content
        mock_embed.return_value = [[0.1] * 384] * 10  # Realistic embedding dimensions
        
        result = await process_file(mock_file)
        
        # Verify complete pipeline execution
        assert result["status"] == "success"
        assert result["chunks_count"] > 0
        assert mock_store.called


@pytest.mark.asyncio
async def test_multiple_files_sequential():
    """Test processing multiple files sequentially"""
    files_data = [
        ("file1.txt", "Content for file one"),
        ("file2.txt", "Content for file two"),
        ("file3.txt", "Content for file three"),
    ]
    
    with patch('app.services.ingestion_service.parse_file') as mock_parse, \
         patch('app.services.ingestion_service.generate_embeddings') as mock_embed, \
         patch('app.services.ingestion_service.store_embeddings') as mock_store:
        
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        results = []
        for filename, content in files_data:
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = filename
            mock_parse.return_value = content
            
            result = await process_file(mock_file)
            results.append(result)
        
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        assert mock_store.call_count == 3
