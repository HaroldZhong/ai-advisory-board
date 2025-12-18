"""
Manual test script for Phase 1: Query Rewriting

Run this with a few test conversations to verify query rewriting works.
"""

async def test_query_rewriting():
    from backend.council import rewrite_query
    
    # Test Case 1: Short follow-up with coreference
    print("=" * 60)
    print("TEST 1: Short follow-up with coreference")
    print("=" * 60)
    
    conversation_history = [
        {"role": "user", "content": "How does RAG work?"},
        {"role": "assistant", "stage3": {
            "response": "RAG (Retrieval-Augmented Generation) combines information retrieval with language generation. It first retrieves relevant documents from a knowledge base..."
        }},
    ]
    
    query = "What about its limitations?"
    rewritten = await rewrite_query(query, conversation_history)
    print(f"Original:  {query}")
    print(f"Rewritten: {rewritten}")
    assert "RAG" in rewritten or "retrieval" in rewritten.lower()
    print("✅ PASS: Coreference resolved\n")
    
    # Test Case 2: Already self-contained (should skip)
    print("=" * 60)
    print("TEST 2: Self-contained query (should skip rewriting)")
    print("=" * 60)
    
    long_query = "Can you explain how vector embeddings work in semantic search systems?"
    rewritten2 = await rewrite_query(long_query, conversation_history)
    print(f"Original:  {long_query}")
    print(f"Rewritten: {rewritten2}")
    assert rewritten2 == long_query  # Should be unchanged
    print("✅ PASS: Skipped rewrite for self-contained query\n")
    
    # Test Case 3: Empty history (should skip)
    print("=" * 60)
    print("TEST 3: No context available (should skip)")
    print("=" * 60)
    
    query3 = "What about Python?"
    rewritten3 = await rewrite_query(query3, [])
    print(f"Original:  {query3}")
    print(f"Rewritten: {rewritten3}")
    assert rewritten3 == query3  # Should be unchanged
    print("✅ PASS: Skipped rewrite when no context\n")
    
    # Test Case 4: Feature flag disabled
    print("=" * 60)
    print("TEST 4: Feature flag disabled")
    print("=" * 60)
    
    from backend import config
    original_flag = config.ENABLE_QUERY_REWRITE
    config.ENABLE_QUERY_REWRITE = False
    
    query4 = "What about its downsides?"
    rewritten4 = await rewrite_query(query4, conversation_history)
    print(f"Original:  {query4}")
    print(f"Rewritten: {rewritten4}")
    assert rewritten4 == query4  # Should be unchanged when disabled
    print("✅ PASS: Feature flag works\n")
    
    # Restore flag
    config.ENABLE_QUERY_REWRITE = original_flag
    
    print("=" * 60)
    print("ALL TESTS PASSED! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_query_rewriting())
