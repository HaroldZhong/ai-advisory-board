import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os

# RAG Configuration
RAG_SIM_THRESHOLD = 0.2  # Lowered from 0.3 for better retrieval
RAG_MAX_TOKENS = 1000

class CouncilRAG:
    def __init__(self, persist_path: str = "./data/chroma_db"):
        """
        Initialize the Council RAG system with ChromaDB.
        """
        # Ensure the directory exists
        os.makedirs(persist_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Use a lightweight, local embedding model
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get the collection with cosine distance
        self.collection = self.client.get_or_create_collection(
            name="council_context",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_session(
        self, 
        conversation_id: str, 
        turn_index: int, 
        user_question: str,
        stage1_results: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        stage3_result: Dict[str, Any]
    ):
        """
        Index a full council session (Stage 1, 2, 3) into ChromaDB.
        Uses upsert behavior with deterministic IDs.
        """
        ids = []
        documents = []
        metadatas = []

        # Helper to format text
        def format_text(text: str) -> str:
            return f"Q: {user_question}\n\nA: {text}"

        # Stage 1: Individual Opinions
        for res in stage1_results:
            model = res['model']
            text = res['response']
            
            doc_id = f"{conversation_id}:{turn_index}:opinion:{model}"
            ids.append(doc_id)
            documents.append(format_text(text))
            metadatas.append({
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "stage": "opinion",
                "model": model
            })

        # Stage 2: Peer Reviews
        for res in stage2_results:
            model = res['model']
            text = res['ranking']  # This contains the critique/ranking text
            
            doc_id = f"{conversation_id}:{turn_index}:review:{model}"
            ids.append(doc_id)
            documents.append(format_text(text))
            metadatas.append({
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "stage": "review",
                "model": model
            })

        # Stage 3: Final Synthesis
        final_text = stage3_result.get('response', '')
        if final_text:
            doc_id = f"{conversation_id}:{turn_index}:synthesis:chairman"
            ids.append(doc_id)
            documents.append(format_text(final_text))
            metadatas.append({
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "stage": "synthesis",
                "model": "chairman"
            })

        # Upsert to ChromaDB
        if ids:
            print(f"[RAG] Indexing {len(ids)} chunks for conversation {conversation_id}, turn {turn_index}")
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def retrieve(self, query: str, conversation_id: str) -> str:
        """
        Retrieve relevant context for a query within a specific conversation.
        Returns a formatted string of context chunks.
        """
        # Query ChromaDB
        print(f"[RAG] Querying for conversation_id: {conversation_id}")
        results = self.collection.query(
            query_texts=[query],
            n_results=5,  # Fetch a few candidates
            where={"conversation_id": conversation_id}
        )

        print(f"[RAG] ChromaDB returned: {len(results['ids'][0]) if results['ids'] and results['ids'][0] else 0} results")
        
        if not results['ids'] or not results['ids'][0]:
            print(f"[RAG] No documents found in ChromaDB for conversation {conversation_id}")
            return ""

        # Process results
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]

        # Combine into a list of dicts for sorting
        chunks = []
        for i in range(len(ids)):
            # Calculate similarity from cosine distance
            similarity = 1 - distances[i]
            
            if similarity >= RAG_SIM_THRESHOLD:
                chunks.append({
                    "id": ids[i],
                    "similarity": similarity,
                    "metadata": metadatas[i],
                    "text": documents[i]
                })

        # Sort by similarity descending
        chunks.sort(key=lambda x: x['similarity'], reverse=True)

        # Build context with token budget
        formatted_context = []
        current_tokens = 0
        
        print(f"[RAG] Query: '{query}'")
        print(f"[RAG] Found {len(chunks)} relevant chunks (threshold {RAG_SIM_THRESHOLD})")

        for chunk in chunks:
            # Approx token count
            tokens = int(len(chunk['text'].split()) * 1.3)
            
            if current_tokens + tokens > RAG_MAX_TOKENS:
                print(f"[RAG] Skipping chunk {chunk['id']} (budget exceeded)")
                continue
                
            current_tokens += tokens
            formatted_context.append(self._format_chunk(chunk))
            print(f"[RAG] Added chunk {chunk['id']} (sim: {chunk['similarity']:.3f}, tokens: {tokens})")

        return "\n\n".join(formatted_context)

    def _format_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Format a single chunk for the LLM context.
        """
        meta = chunk['metadata']
        # Remove the "Q: ... A: " prefix for the context block to save tokens/reduce repetition if desired,
        # OR keep it. The plan said "Prepend user question to indexed text". 
        # The prompt will see "Q: ... A: ...". This is fine.
        
        return f"""[Turn {meta['turn_index']} | Stage {meta['stage']} | Model: {meta['model']}]
{chunk['text']}"""
