"""
OpenRouter PDF Processing Integration.

Provides enhanced PDF extraction using OpenRouter's PDF processing engines:
- pdf-text: Free, good for well-structured PDFs with clear text
- mistral-ocr: Paid ($2/1k pages), best for scanned documents or image-based PDFs

Reference: https://openrouter.ai/docs/guides/overview/multimodal/pdfs
"""

import base64
import httpx
from typing import Dict, Any, Optional, Literal
from .config import OPENROUTER_API_KEY
from .logger import logger

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# PDF Processing engines
PDFEngine = Literal["pdf-text", "mistral-ocr", "native"]

# Pricing per 1000 pages
ENGINE_PRICING = {
    "pdf-text": 0.0,       # Free
    "mistral-ocr": 2.0,    # $2 per 1000 pages
    "native": 0.0,         # Charged as input tokens
}


def estimate_pdf_cost(page_count: int, engine: PDFEngine) -> float:
    """
    Estimate cost for PDF processing.
    
    Args:
        page_count: Number of pages in the PDF
        engine: Processing engine to use
    
    Returns:
        Estimated cost in USD
    """
    price_per_1k = ENGINE_PRICING.get(engine, 0.0)
    return (page_count / 1000) * price_per_1k


async def extract_pdf_with_openrouter(
    pdf_content: bytes,
    filename: str,
    engine: PDFEngine = "pdf-text",
    use_zdr: bool = False,
) -> Dict[str, Any]:
    """
    Extract text from PDF using OpenRouter's PDF processing.
    
    Args:
        pdf_content: Raw PDF bytes
        filename: Original filename
        engine: Processing engine ("pdf-text" or "mistral-ocr")
        use_zdr: Enable Zero Data Retention (privacy mode)
    
    Returns:
        Dict with:
        - status: "success" | "partial" | "failed"
        - text: Extracted text
        - error: Error message if failed
        - usage: Token usage info
        - cost: Estimated cost
        - annotations: File annotations for caching
    """
    if not OPENROUTER_API_KEY:
        return {
            "status": "failed",
            "text": "",
            "error": "OpenRouter API key not configured",
            "usage": {},
            "cost": 0.0,
        }
    
    # Encode PDF to base64
    base64_pdf = base64.b64encode(pdf_content).decode("utf-8")
    data_url = f"data:application/pdf;base64,{base64_pdf}"
    
    # Build request payload
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please extract and return the full text content of this PDF document. Preserve paragraph structure and maintain readability. Include any tables, figures descriptions, and headings."
                },
                {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": data_url
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": "google/gemini-2.5-flash",  # Fast model for extraction
        "messages": messages,
        "plugins": [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": engine
                }
            }
        ]
    }
    
    # Add ZDR if enabled
    if use_zdr:
        payload["provider"] = {"zdr": True}
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://llm-council.local",
        "X-Title": "AI Advisory Board"
    }
    
    logger.info(f"[OPENROUTER-PDF] Processing {filename} with engine={engine}, zdr={use_zdr}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"[OPENROUTER-PDF] API error {response.status_code}: {error_text}")
                return {
                    "status": "failed",
                    "text": "",
                    "error": f"OpenRouter API error: {response.status_code}",
                    "usage": {},
                    "cost": 0.0,
                }
            
            data = response.json()
            
            # Extract response content
            content = ""
            annotations = None
            
            if data.get("choices") and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                annotations = message.get("annotations")
            
            # Get usage info
            usage = data.get("usage", {})
            
            # Estimate page count from content (rough estimate if not available)
            # Average ~3000 chars per page
            estimated_pages = max(1, len(content) // 3000) if content else 1
            cost = estimate_pdf_cost(estimated_pages, engine)
            
            logger.info(f"[OPENROUTER-PDF] Extracted {len(content)} chars, estimated pages: {estimated_pages}")
            
            return {
                "status": "success" if content else "partial",
                "text": content,
                "error": None if content else "No content extracted",
                "usage": usage,
                "cost": cost,
                "annotations": annotations,
            }
            
    except httpx.TimeoutException:
        logger.error(f"[OPENROUTER-PDF] Timeout processing {filename}")
        return {
            "status": "failed",
            "text": "",
            "error": "PDF processing timed out (>120s)",
            "usage": {},
            "cost": 0.0,
        }
    except Exception as e:
        logger.error(f"[OPENROUTER-PDF] Error: {e}")
        return {
            "status": "failed",
            "text": "",
            "error": f"PDF processing failed: {str(e)}",
            "usage": {},
            "cost": 0.0,
        }


def get_engine_recommendation(
    char_count: int,
    empty_page_ratio: float,
    page_count: int = 1
) -> Dict[str, Any]:
    """
    Recommend which PDF engine to use based on local extraction metrics.
    
    Args:
        char_count: Characters extracted locally
        empty_page_ratio: Ratio of empty pages (0.0 - 1.0)
        page_count: Number of pages in PDF
    
    Returns:
        Dict with:
        - needs_enhanced: Whether enhanced extraction is recommended
        - recommended_engine: "pdf-text" or "mistral-ocr"
        - reason: Human-readable explanation
        - estimated_cost: Cost for recommended engine
    """
    # Thresholds from PRD
    MIN_CHARS = 500
    MAX_EMPTY_RATIO = 0.5
    
    needs_enhanced = char_count < MIN_CHARS or empty_page_ratio > MAX_EMPTY_RATIO
    
    if not needs_enhanced:
        return {
            "needs_enhanced": False,
            "recommended_engine": None,
            "reason": "Local extraction successful",
            "estimated_cost": 0.0,
        }
    
    # Determine reason and engine
    if char_count < MIN_CHARS and empty_page_ratio > MAX_EMPTY_RATIO:
        # Likely scanned/image PDF - need OCR
        engine = "mistral-ocr"
        reason = "This PDF appears to be scanned or image-based. OCR extraction is recommended."
    elif empty_page_ratio > MAX_EMPTY_RATIO:
        # Many empty pages - could be complex layout
        engine = "mistral-ocr"
        reason = f"{int(empty_page_ratio * 100)}% of pages appear empty. Enhanced OCR may help extract content."
    else:
        # Low char count but some pages have content - try free engine first
        engine = "pdf-text"
        reason = "Limited text extracted locally. Enhanced text extraction may improve results."
    
    cost = estimate_pdf_cost(page_count, engine)
    
    return {
        "needs_enhanced": True,
        "recommended_engine": engine,
        "reason": reason,
        "estimated_cost": cost,
        "cost_hint": f"${cost:.3f}" if cost > 0 else "Free",
    }
