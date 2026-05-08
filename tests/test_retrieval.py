"""
Comprehensive tests for the retrieval service.
Tests query processing, embedding generation, vector search, and error handling.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.retrieval_service import retrieve
from app.core.exceptions import AppException


# ========================
# Retrieval Tests
# ========================

def test_retrieve_success_default_top_k():
    """Test successful retrieval with default top_k parameter"""
    query = "What is machine learning?"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 3
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection.query.return_value = {
            'documents': [['Machine learning is...', 'AI and ML are...', 'Deep learning is...']],
            'ids': [['id1', 'id2', 'id3']],
            'distances': [[0.1, 0.2, 0.3]],
            'metadatas': [[
                {'filename': 'ml_basics.txt', 'chunk_id': 0},
                {'filename': 'ml_basics.txt', 'chunk_id': 1},
                {'filename': 'ml_basics.txt', 'chunk_id': 2}
            ]]
        }
        
        # Execute
        result = retrieve(query)
        
        # Assertions
        assert len(result) == 3
        assert result[0]['content'] == 'Machine learning is...'
        assert result[0]['id'] == 'id1'
        assert result[0]['distance'] == 0.1
        assert result[0]['metadata']['filename'] == 'ml_basics.txt'
        assert result[0]['metadata']['chunk_id'] == 0
        
        # Verify calls
        mock_embed.assert_called_once_with([query])
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=3
        )


def test_retrieve_success_custom_top_k():
    """Test successful retrieval with custom top_k parameter"""
    query = "Explain neural networks"
    custom_top_k = 5
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection.query.return_value = {
            'documents': [['Neural networks...', 'Deep learning...', 'Backpropagation...', 'Activation...', 'Layers...']],
            'ids': [['id1', 'id2', 'id3', 'id4', 'id5']],
            'distances': [[0.1, 0.2, 0.25, 0.3, 0.35]],
            'metadatas': [[{}, {}, {}, {}, {}]]
        }
        
        # Execute
        result = retrieve(query, top_k=custom_top_k)
        
        # Assertions
        assert len(result) == 5
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5
        )


def test_retrieve_with_top_k_exceeding_max():
    """Test that top_k is capped at RETRIEVAL_MAX_TOP_K"""
    query = "Test query"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection.query.return_value = {
            'documents': [['result'] * 10],
            'ids': [[f'id{i}' for i in range(10)]],
            'distances': [[0.1] * 10],
            'metadatas': [[{}] * 10]
        }
        
        # Execute with top_k > max
        result = retrieve(query, top_k=50)
        
        # Assertions - should be capped at 10
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=10
        )


def test_retrieve_empty_query():
    """Test retrieval with empty query string"""
    with pytest.raises(AppException) as exc_info:
        retrieve("")
    
    assert exc_info.value.status_code == 400
    assert "Query cannot be empty" in str(exc_info.value.message)


def test_retrieve_whitespace_query():
    """Test retrieval with whitespace-only query"""
    with pytest.raises(AppException) as exc_info:
        retrieve("   \n  \t  ")
    
    assert exc_info.value.status_code == 400
    assert "Query cannot be empty" in str(exc_info.value.message)


def test_retrieve_zero_top_k():
    """Test retrieval with top_k = 0"""
    with pytest.raises(AppException) as exc_info:
        retrieve("test query", top_k=0)
    
    assert exc_info.value.status_code == 400
    assert "top_k must be greater than 0" in str(exc_info.value.message)


def test_retrieve_negative_top_k():
    """Test retrieval with negative top_k"""
    with pytest.raises(AppException) as exc_info:
        retrieve("test query", top_k=-5)
    
    assert exc_info.value.status_code == 400
    assert "top_k must be greater than 0" in str(exc_info.value.message)


def test_retrieve_no_results():
    """Test retrieval when no matching documents are found"""
    query = "Very specific query with no matches"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 3
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        # Empty results
        mock_collection.query.return_value = {
            'documents': [[]],
            'ids': [[]],
            'distances': [[]],
            'metadatas': [[]]
        }
        
        # Execute
        result = retrieve(query)
        
        # Assertions
        assert len(result) == 0
        assert result == []


def test_retrieve_without_metadata():
    """Test retrieval when results don't have metadata"""
    query = "Test query"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 2
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        # Results without metadata
        mock_collection.query.return_value = {
            'documents': [['content1', 'content2']],
            'ids': [['id1', 'id2']],
            'distances': [[0.1, 0.2]]
            # No metadatas key
        }
        
        # Execute
        result = retrieve(query)
        
        # Assertions
        assert len(result) == 2
        assert 'metadata' not in result[0]
        assert result[0]['content'] == 'content1'
        assert result[0]['id'] == 'id1'
        assert result[0]['distance'] == 0.1


def test_retrieve_without_distances():
    """Test retrieval when results don't have distance scores"""
    query = "Test query"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 2
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        # Results without distances
        mock_collection.query.return_value = {
            'documents': [['content1', 'content2']],
            'ids': [['id1', 'id2']],
            'metadatas': [[{'source': 'test'}, {'source': 'test'}]]
            # No distances key
        }
        
        # Execute
        result = retrieve(query)
        
        # Assertions
        assert len(result) == 2
        assert result[0]['distance'] is None
        assert result[0]['content'] == 'content1'


def test_retrieve_embedding_generation_error():
    """Test retrieval when embedding generation fails"""
    query = "Test query"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        mock_settings.RETRIEVAL_TOP_K = 3
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.side_effect = Exception("Embedding API error")
        
        with pytest.raises(AppException) as exc_info:
            retrieve(query)
        
        assert exc_info.value.status_code == 500
        assert "Failed to retrieve chunks" in str(exc_info.value.message)


def test_retrieve_database_query_error():
    """Test retrieval when database query fails"""
    query = "Test query"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 3
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        mock_collection.query.side_effect = Exception("Database connection error")
        
        with pytest.raises(AppException) as exc_info:
            retrieve(query)
        
        assert exc_info.value.status_code == 500
        assert "Failed to retrieve chunks" in str(exc_info.value.message)


def test_retrieve_long_query():
    """Test retrieval with a very long query string"""
    query = "What is machine learning? " * 100  # Long query
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 3
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1] * 384]  # Standard embedding size
        
        mock_collection.query.return_value = {
            'documents': [['result1']],
            'ids': [['id1']],
            'distances': [[0.1]],
            'metadatas': [[{}]]
        }
        
        # Execute
        result = retrieve(query)
        
        # Assertions
        assert len(result) == 1
        mock_embed.assert_called_once_with([query])


def test_retrieve_special_characters_in_query():
    """Test retrieval with special characters in query"""
    query = "What is C++ & Python? <script>alert('test')</script>"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_TOP_K = 3
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection.query.return_value = {
            'documents': [['C++ is...']],
            'ids': [['id1']],
            'distances': [[0.1]],
            'metadatas': [[{}]]
        }
        
        # Execute
        result = retrieve(query)
        
        # Assertions
        assert len(result) == 1
        mock_embed.assert_called_once_with([query])


# ========================
# Edge Case Tests
# ========================

def test_retrieve_with_top_k_equals_one():
    """Test retrieval with top_k = 1"""
    query = "Single result query"
    
    with patch('app.services.retrieval_service.generate_embeddings') as mock_embed, \
         patch('app.services.retrieval_service.collection') as mock_collection, \
         patch('app.services.retrieval_service.settings') as mock_settings:
        
        # Setup mocks
        mock_settings.RETRIEVAL_MAX_TOP_K = 10
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        mock_collection.query.return_value = {
            'documents': [['Single result']],
            'ids': [['id1']],
            'distances': [[0.05]],
            'metadatas': [[{'source': 'test.txt'}]]
        }
        
        # Execute
        result = retrieve(query, top_k=1)
        
        # Assertions
        assert len(result) == 1
        assert result[0]['content'] == 'Single result'
        assert result[0]['distance'] == 0.05
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1
        )


def test_retrieve_none_query():
    """Test retrieval with None as query"""
    with pytest.raises(Exception):
        # This should raise an error before validation
        retrieve(None)
