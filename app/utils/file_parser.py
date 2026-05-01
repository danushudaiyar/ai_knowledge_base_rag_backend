# PDF, DOCX, TXT parsing and URL/HTML extraction
from fastapi import UploadFile
import io
import httpx
from bs4 import BeautifulSoup
from typing import Optional
import re
from app.core.logging import logger


def is_url(text: str) -> bool:
    """
    Check if a string is a valid URL
    
    Args:
        text: String to check
    
    Returns:
        True if text is a URL, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text))


async def fetch_url_content(url: str, timeout: int = 30) -> str:
    """
    Fetch content from a URL
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
    
    Returns:
        HTML content as string
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            logger.info(f"Fetched URL: {url}, status: {response.status_code}")
            return response.text
    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching URL {url}: {str(e)}")
        raise ValueError(f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        raise ValueError(f"Failed to fetch URL: {str(e)}")


def extract_text_from_html(html_content: str) -> str:
    """
    Extract readable text from HTML content
    
    Args:
        html_content: HTML content as string
    
    Returns:
        Extracted text with HTML tags removed
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up text: remove extra whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Extracted {len(text)} characters from HTML")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        raise ValueError(f"Failed to parse HTML: {str(e)}")


async def parse_url(url: str) -> str:
    """
    Parse URL and extract text from HTML
    
    Args:
        url: URL to parse
    
    Returns:
        Extracted text content
    """
    logger.info(f"Parsing URL: {url}")
    html_content = await fetch_url_content(url)
    text = extract_text_from_html(html_content)
    return text


async def parse_file(file: UploadFile) -> str:
    """
    Parse uploaded file and extract text content
    
    Args:
        file: UploadFile object from FastAPI
    
    Returns:
        Extracted text content as string
    """
    try:
        # Read file content
        content = await file.read()
        
        # Decode content to text (assuming UTF-8 encoding)
        text = content.decode('utf-8')
        
        # Check if content is HTML
        if text.strip().startswith('<!DOCTYPE') or text.strip().startswith('<html'):
            logger.info(f"Detected HTML content in file: {file.filename}")
            text = extract_text_from_html(text)
        
        logger.info(f"Parsed file: {file.filename}, length: {len(text)} characters")
        return text
        
    except UnicodeDecodeError:
        logger.error(f"Failed to decode file: {file.filename}")
        raise ValueError(f"Unable to decode file {file.filename}. Please ensure it's a text file.")
    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {str(e)}")
        raise
