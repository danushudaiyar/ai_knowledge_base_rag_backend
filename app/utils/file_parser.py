# PDF, DOCX, TXT parsing
from fastapi import UploadFile
import io
from app.core.logging import logger


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
        
        logger.info(f"Parsed file: {file.filename}, length: {len(text)} characters")
        return text
        
    except UnicodeDecodeError:
        logger.error(f"Failed to decode file: {file.filename}")
        raise ValueError(f"Unable to decode file {file.filename}. Please ensure it's a text file.")
    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {str(e)}")
        raise
