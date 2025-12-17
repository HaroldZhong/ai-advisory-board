from typing import Dict, Any, Callable, List, Optional
from .types import ToolDefinition

class ToolRegistry:
    """
    Central registry for all available tools.
    """
    _tools: Dict[str, ToolDefinition] = {}
    _implementations: Dict[str, Callable] = {}

    @classmethod
    def register(cls, 
                 name: str, 
                 description: str, 
                 args_schema: Dict[str, Any], 
                 result_schema: Optional[Dict[str, Any]] = None,
                 examples: List[str] = None):
        """
        Decorator to register a tool implementation.
        """
        def decorator(func: Callable):
            definition = ToolDefinition(
                name=name,
                description=description,
                args_schema=args_schema,
                result_schema=result_schema,
                examples=examples or []
            )
            cls._tools[name] = definition
            cls._implementations[name] = func
            return func
        return decorator

    @classmethod
    def get_implementation(cls, name: str) -> Optional[Callable]:
        """Get the executable function for a tool."""
        return cls._implementations.get(name)

    @classmethod
    def get_definition(cls, name: str) -> Optional[ToolDefinition]:
        """Get the metadata for a tool."""
        return cls._tools.get(name)

    @classmethod
    def list_tools(cls) -> List[ToolDefinition]:
        """List all registered tools."""
        return list(cls._tools.values())

    @classmethod
    def to_prompt_format(cls) -> str:
        """
        Render available tools for the Steward prompt.
        """
        if not cls._tools:
            return "No tools available."
        
        lines = ["Available Tools:"]
        for tool in cls._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  Arguments: {tool.args_schema}")
            if tool.examples:
                lines.append(f"  Example intents: {', '.join(tool.examples)}")
            lines.append("")
        return "\n".join(lines)
