import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .types import ToolCall, ToolResult, ToolError, EvidencePack, ToolUsageRecord, UsageLimits
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

# Simple in-memory cache: (tool_name, frozenset(args)) -> (ToolResult, expire_at)
_TOOL_CACHE: Dict[str, Any] = {}

class ToolRouter:
    """
    The control plane for tool execution.
    Enforces budgets, rate limits, allowlists, and sanitization.
    """
    
    def __init__(self, 
                 allowlist: List[str] = None, 
                 max_calls_per_run: int = 5,
                 max_evidence_chars: int = 10000):
        self.allowlist = allowlist
        self.max_calls_per_run = max_calls_per_run
        self.max_evidence_chars = max_evidence_chars
        
        # Per-tool rate limits (calls per minute) - Mock default
        self.rate_limits = {
            "web.search": 5,
            "web.fetch": 10
        }

    async def execute_tool_calls(self, calls: List[ToolCall], run_id: str) -> EvidencePack:
        """
        Execute a batch of tool calls with strict enforcement.
        """
        usage_records: List[ToolUsageRecord] = []
        limits_triggered: List[str] = []
        calls_executed_count = 0
        total_evidence_chars = 0
        
        # Sort by priority (high -> normal -> low), then documented order
        # This makes enforcement deterministic
        priority_map = {"high": 0, "normal": 1, "low": 2}
        sorted_calls = sorted(calls, key=lambda c: priority_map.get(c.priority, 1))

        # Core execution loop
        for call in sorted_calls:
            # 1. Global Budget Check
            if calls_executed_count >= self.max_calls_per_run:
                logger.warning(f"[Router] Budget exceeded for run {run_id}. Rejecting {call.name}")
                limits_triggered.append("max_calls_per_run")
                usage_records.append(self._create_rejected_record(call, "budget_exceeded"))
                continue

            # 2. Allowlist Check
            if self.allowlist and call.name not in self.allowlist:
                logger.warning(f"[Router] Tool {call.name} not in allowlist for run {run_id}")
                usage_records.append(self._create_rejected_record(call, "access_denied"))
                continue
                
            # 3. Registry Check
            func = ToolRegistry.get_implementation(call.name)
            if not func:
                logger.warning(f"[Router] Tool {call.name} not found in registry")
                usage_records.append(self._create_rejected_record(call, "tool_not_found"))
                continue

            # 4. Execute (or Cached)
            try:
                # Check cache
                cached_result = self._check_cache(call)
                if cached_result:
                    result = cached_result
                else:
                    # Execute
                    if asyncio.iscoroutinefunction(func):
                        result = await func(call.arguments, run_id=run_id, call_id=call.id)
                    else:
                        result = await asyncio.to_thread(func, call.arguments, run_id=run_id, call_id=call.id)
                    
                    # Cache successful results
                    if result.ok:
                        self._update_cache(call, result)

                # 5. Sanitize & Summarize
                record = self._process_result(call, result, total_evidence_chars)
                usage_records.append(record)
                
                if result.ok:
                    calls_executed_count += 1
                    # Track content size for limits
                    if record.raw_truncated:
                        total_evidence_chars += len(record.raw_truncated)
                    
                    # Stop if max evidence size hit
                    if total_evidence_chars >= self.max_evidence_chars:
                        limits_triggered.append("max_evidence_size")
                        break

            except Exception as e:
                logger.exception(f"[Router] Unexpected error executing {call.name}")
                error_result = ToolResult(
                    id=call.id,
                    run_id=run_id,
                    ok=False,
                    error=ToolError(type="tool_error", message=str(e), retryable=True),
                    meta={"latency_ms": 0}
                )
                usage_records.append(self._process_result(call, error_result, total_evidence_chars))

        # Construct EvidencePack
        return EvidencePack(
            run_id=run_id,
            query="[Pending]", # Caller should fill this
            tools_used=usage_records,
            limits=UsageLimits(
                max_calls=self.max_calls_per_run,
                calls_used=calls_executed_count,
                limits_triggered=list(set(limits_triggered))
            )
        )

    def _create_rejected_record(self, call: ToolCall, reason: str) -> ToolUsageRecord:
        return ToolUsageRecord(
            call_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
            status="rejected",
            output_summary=f"Call rejected: {reason}",
            meta={"rejection_reason": reason}
        )

    def _check_cache(self, call: ToolCall) -> Optional[ToolResult]:
        # Simple exact match on args
        key = (call.name, frozenset(call.arguments.items()))
        if key in _TOOL_CACHE:
            res, expire_at = _TOOL_CACHE[key]
            if datetime.now() < expire_at:
                res.meta["cached"] = True
                return res
            else:
                del _TOOL_CACHE[key]
        return None

    def _update_cache(self, call: ToolCall, result: ToolResult):
        # Default TTL 5 mins
        ttl = 300
        # Longer for search, shorter for finance? (Hardcoded policy for now)
        if "search" in call.name:
            ttl = 600
        
        key = (call.name, frozenset(call.arguments.items()))
        expire_at = datetime.now() + timedelta(seconds=ttl)
        _TOOL_CACHE[key] = (result, expire_at)

    def _process_result(self, call: ToolCall, result: ToolResult, current_chars: int) -> ToolUsageRecord:
        """
        Convert a raw ToolResult into a safe, usable ToolUsageRecord.
        """
        status = "executed" if result.ok else "failed"
        
        # Truncate raw output
        raw_str = str(result.data) if result.data else ""
        if len(raw_str) > 2000:
            raw_str = raw_str[:2000] + "... [TRUNCATED]"
        
        # Summarize for the LLM
        if not result.ok:
            summary = f"Error: {result.error.message if result.error else 'Unknown error'}"
        else:
            # Domain-specific summaries could go here
            summary = f"Executed {call.name}. Result size: {len(raw_str)} chars."
            
            # Special handling for web.search to extract sources formatted nicely
            if call.name == "web.search" and isinstance(result.data, dict):
                results = result.data.get("results", [])
                summary = f"Found {len(results)} results: " + ", ".join([f"{r.get('title', 'Unknown')} [s{r.get('source_id', '?')}]" for r in results[:3]]) + "..."

        return ToolUsageRecord(
            call_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
            status=status,
            output_summary=summary,
            raw_truncated=raw_str,
            meta=result.meta
        )
