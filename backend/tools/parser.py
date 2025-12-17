import json
import logging
import re
from typing import Dict, Any, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class StewardAction(BaseModel):
    """
    Expected structure from the Tool Steward.
    """
    action: str  # "use_tools" or "no_tools"
    reason: Optional[str] = None
    calls: Optional[list] = None  # List of dicts

class ToolParser:
    """
    Robust parser for extracting JSON from LLM responses.
    Handles markdown blocks, extra text, and common malformations.
    """

    @staticmethod
    def parse_steward_output(text: str) -> Dict[str, Any]:
        """
        Extract and validate the Steward's JSON output.
        Returns a clean dictionary or a fallback 'no_tools' action.
        """
        logger.debug(f"[Parser] Parsing text: {text[:200]}...")
        
        try:
            # 1. Cleaning: Remove markdown code blocks
            clean_text = ToolParser._strip_markdown(text)
            
            # 2. Extraction: Find the first/largest JSON object
            json_str = ToolParser._extract_json_string(clean_text)
            
            if not json_str:
                logger.warning("[Parser] No JSON found in response.")
                return {"action": "no_tools", "reason": "output_parsing_failed"}

            # 3. Parsing
            data = json.loads(json_str)
            
            # 4. Validation (Basic Structure)
            # We don't enforce strict schema here to allow for "repair" logic if needed,
            # but we at least check for 'action'.
            if "action" not in data:
                 # Heuristic: if 'calls' exists, assume 'use_tools'
                 if "calls" in data and isinstance(data["calls"], list):
                     data["action"] = "use_tools"
                 else:
                     data["action"] = "no_tools"
                     data["reason"] = "missing_action_field"
            
            return data

        except json.JSONDecodeError as e:
            logger.error(f"[Parser] Invalid JSON: {e}")
            return {"action": "no_tools", "reason": "json_decode_error"}
        except Exception as e:
            logger.exception(f"[Parser] Unexpected error: {e}")
            return {"action": "no_tools", "reason": "parser_unexpected_error"}

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove ```json ... ``` blocks."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    @staticmethod
    def _extract_json_string(text: str) -> Optional[str]:
        """
        Find the substring that looks like the outermost JSON object.
        """
        text = text.strip()
        
        # Simple heuristic: find first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        
        return None
