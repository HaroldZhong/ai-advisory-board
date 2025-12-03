"""
File processing logic for the LLM Council.
Handles text extraction from various file types and image description via vision models.
"""

import io
from typing import Dict, Any, Optional
from fastapi import UploadFile
import pypdf
import docx
from .openrouter import query_model
from .config import OPENROUTER_API_KEY

# Limits
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
MAX_FILE_SIZE_MB = MAX_FILE_SIZE / (1024 * 1024)
MAX_TEXT_LENGTH = 50000  # 50k characters

async def extract_text_from_file(file: UploadFile) -> Dict[str, Any]:
    """
    Extract text from an uploaded file.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Dict with keys:
        - text: The extracted text (or empty string if failed)
        - truncated: Boolean indicating if text was truncated
        - error: Error message if something went wrong, else None
    """
    filename = file.filename
    content_type = file.content_type
    
    # 1. Check file size
    # Note: UploadFile.size might not be available until read, 
    # but we can check the read content length.
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        return {
            "text": "",
            "truncated": False,
            "error": f"File too large ({file_size / 1024 / 1024:.1f}MB). Max size is {MAX_FILE_SIZE_MB:.0f}MB."
        }
        
    # Reset file cursor for processing
    file.file.seek(0)
    
    try:
        text = ""
        
        # 2. Process based on content type
        # Normalize content type based on extension if generic
        if content_type == "application/octet-stream" or not content_type:
            if filename.lower().endswith(".pdf"):
                content_type = "application/pdf"
            elif filename.lower().endswith(".docx"):
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif filename.lower().endswith(".txt") or filename.lower().endswith(".md"):
                content_type = "text/plain"
            elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
                content_type = "image/jpeg"
            elif filename.lower().endswith(".png"):
                content_type = "image/png"

        if content_type == "application/pdf":
            text = _extract_from_pdf(content)
            
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = _extract_from_docx(content)
            
        elif content_type.startswith("text/") or filename.endswith(".md") or filename.endswith(".txt"):
            # Handle text files (including markdown)
            text = content.decode("utf-8", errors="replace")
            
        elif content_type.startswith("image/"):
            # Handle images via Vision Model
            description = await _describe_image(content, content_type)
            if description:
                text = f"[Image Description: {description}]"
            else:
                return {
                    "text": "",
                    "truncated": False,
                    "error": "Failed to generate image description."
                }
                
        else:
            return {
                "text": "",
                "truncated": False,
                "error": f"Unsupported file type: {content_type}. Supported: PDF, DOCX, TXT, MD, Images."
            }
            
        # 3. Check text length and truncate if needed
        truncated = False
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            truncated = True
            
        return {
            "text": text,
            "truncated": truncated,
            "error": None
        }
        
    except Exception as e:
        return {
            "text": "",
            "truncated": False,
            "error": f"Failed to process file: {str(e)}"
        }

def _extract_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    pdf_file = io.BytesIO(content)
    reader = pypdf.PdfReader(pdf_file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def _extract_from_docx(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    docx_file = io.BytesIO(content)
    doc = docx.Document(docx_file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

async def _describe_image(content: bytes, content_type: str) -> Optional[str]:
    """
    Generate a description for an image using Gemini 2.5 Flash via OpenRouter.
    """
    import base64
    
    # Encode image to base64
    base64_image = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{content_type};base64,{base64_image}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in detail. Focus on text, diagrams, and key visual elements that would be relevant for analysis."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
    
    # Use Gemini 2.5 Flash for vision (fast, cheap, good vision)
    # Note: We need to ensure query_model supports the content list format
    # The current query_model implementation in openrouter.py takes List[Dict[str, str]]
    # We might need to update it or handle the payload construction manually here if query_model is too strict.
    # Let's check openrouter.py again. It types messages as List[Dict[str, str]] but passes it directly to json payload.
    # So as long as the dict structure is valid JSON for OpenRouter, it should work.
    
    response = await query_model("google/gemini-2.5-flash", messages, timeout=30.0)
    
    if response and response.get("content"):
        return response["content"]
    
    return None
