import pytest
import asyncio
import sys
import os

# Ensure backend matches the structure expected by imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.tools.types import ToolCall, ToolResult
from backend.tools.registry import ToolRegistry
from backend.tools.router import ToolRouter
from backend.tools.parser import ToolParser
from backend.tools.definitions.web import web_fetch

# -------------------------------------------------------------------------
# Test 1: Parser Robustness
# -------------------------------------------------------------------------
def test_parser_robustness():
    parser = ToolParser()
    
    # Case 1: Clean JSON
    clean = '{"action": "use_tools", "calls": [{"name": "web.search"}]}'
    assert parser.parse_steward_output(clean)["action"] == "use_tools"

    # Case 2: Markdown wrapped
    markdown = '```json\n{"action": "no_tools", "reason": "logic check"}\n```'
    assert parser.parse_steward_output(markdown)["action"] == "no_tools"

    # Case 3: Extra text noise
    noisy = 'Sure, I can help using tools.\n\n{"action": "use_tools", "calls": []}\n\nHope that helps.'
    res = parser.parse_steward_output(noisy)
    assert res["action"] == "use_tools"

    # Case 4: Broken JSON (fallback)
    broken = '{"action": "use_tools", "calls": [... incomplete'
    res = parser.parse_steward_output(broken)
    assert res["action"] == "no_tools"
    # When no JSON is extracted, it returns output_parsing_failed
    assert res["reason"] == "output_parsing_failed"

# -------------------------------------------------------------------------
# Test 2: Router Determinism & Budget
# -------------------------------------------------------------------------
def test_router_determinism():
    async def _run():
        # Setup: Allowlist only 'web.search'
        router = ToolRouter(allowlist=["web.search"], max_calls_per_run=2)
        
        calls = [
            ToolCall(run_id="test", name="web.search", arguments={"q": "A"}, priority="low", purpose="p", requested_by="check"),
            ToolCall(run_id="test", name="web.search", arguments={"q": "B"}, priority="high", purpose="p", requested_by="check"),
            ToolCall(run_id="test", name="web.search", arguments={"q": "C"}, priority="normal", purpose="p", requested_by="check"),
        ]
        
        # Expected execution order: B (high) -> C (normal) -> A (low/budget exceeded)
        pack = await router.execute_tool_calls(calls, run_id="test1")
        
        executed = [t for t in pack.tools_used if t.status == "executed"]
        rejected = [t for t in pack.tools_used if t.status == "rejected"]
        
        executed_qs = [t.arguments["q"] for t in executed]
        rejected_qs = [t.arguments["q"] for t in rejected]
        
        print(f"Executed: {executed_qs}")
        print(f"Rejected: {rejected_qs}")

        assert len(executed) == 2
        assert len(rejected) == 1
        
        # Verify Order: High prio executed first
        assert executed[0].arguments["q"] == "B"
        assert executed[1].arguments["q"] == "C"
        
        # Verify Rejection
        assert rejected[0].arguments["q"] == "A"
        assert rejected[0].meta["rejection_reason"] == "budget_exceeded"

    asyncio.run(_run())

# -------------------------------------------------------------------------
# Test 3: Safety Limits (Web Fetch)
# -------------------------------------------------------------------------
def test_web_fetch_safety():
    async def _run():
        # 1. Block non-http schemas
        res = await web_fetch({"url": "ftp://malicious.com"}, "run1", "call1")
        assert not res.ok
        assert "validation_error" in res.error.type

        # 2. Sanitization (Mock)
        res = await web_fetch({"url": "http://unsafe.com"}, "run1", "call1")
        assert res.ok
        if "<script>" in str(res.data):
            assert "[SCRIPT REMOVED]" in str(res.data) # Our mock impl does this check

    asyncio.run(_run())

# -------------------------------------------------------------------------
# Test 4: Registry & Prompt
# -------------------------------------------------------------------------
def test_registry_listing():
    prompt = ToolRegistry.to_prompt_format()
    assert "web.search" in prompt
    assert "web.fetch" in prompt
    assert "Search the web" in prompt # Description check

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
