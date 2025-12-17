import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

# -------------------------------------------------------------------------
# Enums and Types
# -------------------------------------------------------------------------

ToolIntent = Literal["search", "fetch", "calculate", "verify", "other"]
ToolPriority = Literal["high", "normal", "low"]
ErrorType = Literal["validation_error", "rate_limited", "budget_exceeded", "tool_error", "timeout", "access_denied", "parse_error"]
ConfidenceBasis = Literal["heuristic", "model", "hybrid"]

# -------------------------------------------------------------------------
# Tool Call & Result
# -------------------------------------------------------------------------

class ToolCall(BaseModel):
    """
    A unified contract for a requested tool execution.
    Includes correlation IDs and intent for logging/enforcement.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    name: str
    arguments: Dict[str, Any]
    intent: ToolIntent = "other"
    purpose: str = Field(description="Short explanation of why this tool is needed")
    priority: ToolPriority = "normal"
    requested_by: str = Field(description="Model ID of the steward")

class ToolError(BaseModel):
    """
    Structured error information for tool failures.
    """
    type: ErrorType
    message: str
    retryable: bool = False
    detail: Optional[str] = None

class ToolResult(BaseModel):
    """
    Normalized result from any tool execution.
    Guarantees 'ok' status and structured data/error.
    """
    id: str  # Matches ToolCall.id
    run_id: str
    ok: bool
    data: Optional[Any] = None
    error: Optional[ToolError] = None
    schema_version: str = "1.0"
    meta: Dict[str, Any] = Field(default_factory=dict)
    # meta headers: latency_ms, cached, rate_limited, source_id (optional)

# -------------------------------------------------------------------------
# Evidence Pack (The Output)
# -------------------------------------------------------------------------

class EvidenceSource(BaseModel):
    """
    A specific cited source within a tool result.
    """
    source_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    fetched_at: str

class UsageLimits(BaseModel):
    """
    Status of budget enforcement for this run.
    """
    max_calls: int = 0
    calls_used: int = 0
    limits_triggered: List[str] = Field(default_factory=list)

class ToolUsageRecord(BaseModel):
    """
    A record of a tool execution for the evidence pack.
    Includes both raw/truncated output and a rendered summary.
    """
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: Literal["executed", "rejected", "failed"]
    output_summary: str = Field(description="Rendered summary with [sID] citations")
    raw_truncated: Optional[str] = Field(description="Truncated raw JSON for debugging", default=None)
    sources: List[EvidenceSource] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

class KeyFact(BaseModel):
    """
    A specific fact extracted or synthesized from tools.
    """
    fact: str
    source_id: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_basis: ConfidenceBasis = "heuristic"

class EvidencePack(BaseModel):
    """
    The complete package of evidence gathered by the Steward.
    This is the interface between Stage 0 (Steward) and Stage 1 (Council).
    """
    run_id: str
    query: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_version: str = "1.0"
    
    tools_used: List[ToolUsageRecord] = Field(default_factory=list)
    key_facts: List[KeyFact] = Field(default_factory=list)
    limits: UsageLimits = Field(default_factory=UsageLimits)
    open_questions: List[str] = Field(default_factory=list)

# -------------------------------------------------------------------------
# Registry Types
# -------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    """
    Metadata for a registered tool.
    """
    name: str
    description: str
    args_schema: Dict[str, Any]  # JSON Schema
    result_schema: Optional[Dict[str, Any]] = None
    examples: List[str] = Field(default_factory=list)
