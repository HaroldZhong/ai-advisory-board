"""
Smoke test for Phase 1 Step 2: Enhanced Metadata

Run this after a couple of council sessions to verify metadata is being indexed correctly.
"""

async def smoke_test_metadata():
    """
    Inspect ChromaDB collection to verify enhanced metadata is stored correctly.
    """
    print("="  * 60)
    print("SMOKE TEST: Enhanced Metadata in ChromaDB")
    print("=" * 60)
    
    from backend.rag import CouncilRAG
    
    # Initialize RAG
    rag = CouncilRAG()
    
    if not rag.enabled:
        print("❌ RAG is disabled, cannot run smoke test")
        return
    
    # Get a sample of indexed documents
    print("\nFetching sample documents from ChromaDB...")
    data = rag.collection.get(
        limit=5,
        include=["metadatas", "documents"]
    )
    
    if not data or not data.get("ids"):
        print("⚠️  No documents found in ChromaDB. Run a council session first.")
        return
    
    print(f"\n✅ Found {len(data['ids'])} documents\n")
    
    # Check metadata fields
    print("Checking metadata fields:")
    print("-" * 60)
    
    expected_fields = [
        "conversation_id",
        "turn_index",
        "stage",
        "model",
        "topics",  # NEW
        "avg_rank",  # NEW
        "consensus_score",  # NEW
        "timestamp",  # NEW
    ]
    
    for i, (doc_id, metadata) in enumerate(zip(data["ids"], data["metadatas"])):
        print(f"\nDocument {i+1}: {doc_id}")
        print(f"  Stage: {metadata.get('stage')}")
        print(f"  Model: {metadata.get('model')}")
        print(f"  Topics: {metadata.get('topics')}")
        print(f"  Avg Rank: {metadata.get('avg_rank')}")
        print(f"  Consensus Score: {metadata.get('consensus_score')}")
        print(f"  Timestamp: {metadata.get('timestamp')}")
        
        # Check all expected fields present
        missing = [field for field in expected_fields if field not in metadata]
        if missing:
            print(f"  ❌ MISSING FIELDS: {missing}")
        else:
            print(f"  ✅ All fields present")
        
        # Validate topics is JSON
        topics_str = metadata.get("topics")
        if topics_str:
            try:
                import json
                topics = json.loads(topics_str)
                if isinstance(topics, list):
                    print(f"  ✅ Topics is valid JSON list: {topics}")
                else:
                    print(f"  ❌ Topics is not a list: {topics}")
            except:
                print(f"  ❌ Topics is not valid JSON: {topics_str}")
        
        # Validate numeric fields
        avg_rank = metadata.get("avg_rank")
        if avg_rank is not None:
            if isinstance(avg_rank, (int, float)) and avg_rank >= 0:
                print(f"  ✅ avg_rank is valid: {avg_rank}")
            else:
                print(f"  ❌ avg_rank is invalid: {avg_rank}")
        
        consensus = metadata.get("consensus_score")
        if consensus is not None:
            if isinstance(consensus, (int, float)) and 0 <= consensus <= 1:
                print(f"  ✅ consensus_score is valid: {consensus}")
            else:
                print(f"  ❌ consensus_score is invalid: {consensus}")
    
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run a few more council sessions")
    print("2. Verify topics look reasonable")
    print("3. Check that avg_rank and consensus_score make sense")
    print("4. If all looks good, Step 2 is DONE ✅")


if __name__ == "__main__":
    import asyncio
    asyncio.run(smoke_test_metadata())
