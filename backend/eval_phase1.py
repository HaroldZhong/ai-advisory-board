"""
Minimal eval script for Phase 1 features.

This is a smoke test, not a benchmark. Tests:
- Query rewriting fires when it should
- Hybrid retrieval hits the right stuff
- Confidence is always present and roughly sensible
"""

import asyncio
from typing import List, Dict, Any

from backend.council import (
    rewrite_query,
    calculate_quality_metrics,
    compute_overall_confidence,
    run_full_council,
)
from backend.rag import CouncilRAG
from backend.logger import logger


EVAL_QUERIES = [
    # Coreference tests
    {
        "category": "coreference",
        "setup": ["How does RAG work?"],
        "follow_up": "What about its limitations?",
    },
    {
        "category": "coreference",
        "setup": ["Tell me what BM25 is."],
        "follow_up": "How does it compare to dense retrieval?",
    },

    # Keyword heavy (test BM25)
    {
        "category": "keyword",
        "setup": ["Explain BM25 in this council system."],
        "follow_up": "What did GPT-5.1 say about BM25?",
        "expect_keywords": ["GPT-5.1", "BM25"],
    },

    # Semantic (test dense)
    {
        "category": "semantic",
        "setup": ["How does retrieval work in this architecture?"],
        "follow_up": "How does it find relevant past turns?",
        "expect_keywords": ["retrieval", "context", "RAG"],
    },

    # Confidence
    {
        "category": "confidence",
        "setup": ["Summarize how RAG works."],
        "follow_up": None,
    },
    {
        "category": "confidence",
        "setup": ["Is Python better than JavaScript for everything?"],
        "follow_up": None,
    },
]


async def run_eval():
    """Run minimal Phase 1 evaluation."""
    rag = CouncilRAG()
    results = {
        "coreference_pass": 0,
        "coreference_total": 0,
        "keyword_pass": 0,
        "keyword_total": 0,
        "semantic_pass": 0,
        "semantic_total": 0,
        "confidence_present": 0,
        "confidence_total": 0,
    }

    conversation_id = "eval_conv"

    print("=" * 60)
    print("PHASE 1 EVALUATION - Smoke Tests")
    print("=" * 60)

    for i, case in enumerate(EVAL_QUERIES):
        category = case["category"]
        logger.info("[EVAL] Case %d category=%s", i, category)
        print(f"\n[Case {i+1}] Category: {category}")

        # Simple in-memory conversation history
        history: List[Dict[str, Any]] = []

        # Run setup turns as full council
        for setup_q in case["setup"]:
            print(f"  Setup: {setup_q}")
            stage1, stage2, stage3, meta = await run_full_council(setup_q)
            quality = calculate_quality_metrics(stage2, meta["label_to_model"])
            
            history.append({"role": "user", "content": setup_q})
            history.append({"role": "assistant", "stage3": stage3})

        # Build query to test
        if case.get("follow_up"):
            user_q = case["follow_up"]
            print(f"  Follow-up: {user_q}")
            rewritten = await rewrite_query(user_q, history)
            logger.info("[EVAL] rewritten=%r", rewritten)
            print(f"  Rewritten: {rewritten}")
            query_for_rag = rewritten
        else:
            user_q = case["setup"][-1]
            query_for_rag = user_q

        # Retrieve context
        context = rag.retrieve(query_for_rag, conversation_id=conversation_id)

        # Simple checks
        if category == "coreference":
            results["coreference_total"] += 1
            if "rag" in query_for_rag.lower() or "bm25" in query_for_rag.lower():
                results["coreference_pass"] += 1
                print("  ✅ Coreference resolved")
            else:
                print("  ❌ Coreference NOT resolved")

        if category == "keyword":
            results["keyword_total"] += 1
            kws = case.get("expect_keywords", [])
            found = all(kw.lower() in context.lower() for kw in kws)
            if found:
                results["keyword_pass"] += 1
                print(f"  ✅ Keywords found: {kws}")
            else:
                print(f"  ❌ Keywords NOT found: {kws}")

        if category == "semantic":
            results["semantic_total"] += 1
            kws = case.get("expect_keywords", [])
            found = any(kw.lower() in context.lower() for kw in kws)
            if found:
                results["semantic_pass"] += 1
                print(f"  ✅ Semantic match found")
            else:
                print(f"  ❌ Semantic match NOT found")

        if category == "confidence":
            # Run a full council to check confidence
            stage1, stage2, stage3, meta = await run_full_council(user_q)
            confidence = stage3.get("confidence")
            results["confidence_total"] += 1
            if confidence in ("HIGH", "MEDIUM", "LOW", "UNKNOWN"):
                results["confidence_present"] += 1
                print(f"  ✅ Confidence: {confidence}")
            else:
                print(f"  ❌ Confidence missing or invalid: {confidence}")
            logger.info("[EVAL] confidence=%s", confidence)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Coreference:       {results['coreference_pass']}/{results['coreference_total']}")
    print(f"Keyword retrieval: {results['keyword_pass']}/{results['keyword_total']}")
    print(f"Semantic retrieval: {results['semantic_pass']}/{results['semantic_total']}")
    print(f"Confidence present: {results['confidence_present']}/{results['confidence_total']}")
    
    print("\nTARGETS:")
    print("- Coreference: at least 3/5")
    print("- Keyword: at least 3/5")
    print("- Semantic: at least 3/5")
    print("- Confidence: 100%")
    
    # Basic pass/fail
    passed = (
        results["coreference_pass"] >= 3 and
        results["confidence_present"] == results["confidence_total"]
    )
    
    if passed:
        print("\n✅ PHASE 1 SMOKE TEST: PASSED")
    else:
        print("\n⚠️  PHASE 1 SMOKE TEST: NEEDS ATTENTION")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_eval())
