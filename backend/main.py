"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings, chat_with_chairman
from .rag import CouncilRAG
from .rag import CouncilRAG
from .file_processing import extract_text_from_file
from .logger import logger

# Initialize RAG system
rag_system = CouncilRAG()

def get_turn_index(conversation: Dict[str, Any]) -> int:
    """Count the number of completed Council turns (messages with stage3)."""
    count = 0
    for msg in conversation.get("messages", []):
        if msg.get("role") == "assistant" and "stage3" in msg:
            count += 1
    return count

def calculate_cost(usage: Dict[str, int], model_id: str) -> float:
    """Calculate cost based on usage and model pricing."""
    if not usage:
        return 0.0
    
    from .config import AVAILABLE_MODELS
    model_config = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    if not model_config:
        return 0.0
    
    pricing = model_config.get('pricing', {})
    input_price = pricing.get('input', 0.0)
    output_price = pricing.get('output', 0.0)
    
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    
    cost = (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price
    return cost

app = FastAPI(title="LLM Council API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    topic: str = "New Conversation"
    council_members: List[str] = None
    chairman_model: str = None


@app.post("/api/conversations")
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    metadata = {}
    
    # Validate council members
    if request.council_members:
        from .config import AVAILABLE_MODELS
        valid_models = {m['id'] for m in AVAILABLE_MODELS}
        invalid = [m for m in request.council_members if m not in valid_models]
        if invalid:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid council models: {invalid}"
            )
        metadata["council_models"] = request.council_members
        
    # Validate chairman model
    if request.chairman_model:
        from .config import AVAILABLE_MODELS
        valid_models = {m['id'] for m in AVAILABLE_MODELS}
        if request.chairman_model not in valid_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid chairman model: {request.chairman_model}"
            )
        metadata["chairman_model"] = request.chairman_model
        
    conversation = storage.create_conversation(conversation_id, metadata)
    return conversation


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    mode: str = "auto"  # "auto", "council", or "chat"


class ConversationMetadata(BaseModel):
    """Conversation metadata for list view."""
    id: str
    created_at: str
    title: str
    message_count: int


class Conversation(BaseModel):
    """Full conversation with all messages."""
    id: str
    created_at: str
    title: str
    messages: List[Dict[str, Any]]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "LLM Council API"}


@app.get("/api/models")
async def get_models():
    """Get list of available models with pricing and capabilities."""
    from .config import AVAILABLE_MODELS
    return {"models": AVAILABLE_MODELS}


@app.get("/api/analytics")
async def get_analytics_data():
    """Get model performance analytics."""
    from .analytics import get_analytics
    return get_analytics()





@app.get("/api/conversations")
async def list_conversations():
    """List all conversations."""
    return storage.list_conversations()


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with all its messages."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process OR chat with chairman.
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Determine mode
    is_first_message = len(conversation["messages"]) == 0
    mode = request.mode
    
    if mode == "auto":
        mode = "council" if is_first_message else "chat"

    # Add user message
    storage.add_user_message(conversation_id, request.content)

    # If this is the first message, generate a title
    if is_first_message:
        title = await generate_conversation_title(request.content)
        storage.update_conversation_title(conversation_id, title)

    # Get model configuration from conversation metadata
    metadata = conversation.get("metadata", {})
    council_models = metadata.get("council_models")
    chairman_model = metadata.get("chairman_model")

    if mode == "council":
        # Run the 3-stage council process
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
            request.content,
            council_models=council_models,
            chairman_model=chairman_model
        )

        # Add assistant message with all stages
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result
        )

        # Index the session for RAG with enhanced metadata
        logger.info("[PHASE1] Indexing turn %d for conversation %s", turn_index, conversation_id)
        
        # Extract topics from question + final answer
        from .council import extract_topics, calculate_quality_metrics
        combined_text = request.content + " " + stage3_result.get('response', '')
        topics = await extract_topics(combined_text, max_topics=3)
        
        # Calculate quality metrics from Stage 2 rankings
        quality_metrics = calculate_quality_metrics(
            stage2_results=stage2_results,
            label_to_model=metadata["label_to_model"],
        )
        
        # Index session with enhanced metadata
        updated_conversation = storage.get_conversation(conversation_id)
        turn_index = get_turn_index(updated_conversation) - 1
        
        rag_system.index_session(
            conversation_id,
            turn_index,
            request.content,
            stage1_results,
            stage2_results,
            stage3_result,
            topics,
            quality_metrics,
        )
        
        # Refresh hybrid index after indexing
        rag_system.refresh_hybrid_index()

        # Return the complete response with metadata
        return {
            "type": "council",
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": stage3_result,
            "metadata": metadata
        }
    else:
        # Chat with Chairman
        # Reload conversation to get the latest user message we just added
        conversation = storage.get_conversation(conversation_id)
        
        # PHASE 1: Rewrite query for better RAG retrieval
        from .council import rewrite_query
        rewritten_query = await rewrite_query(
            request.content,
            conversation["messages"]
        )
        
        # Retrieve context via RAG (using rewritten query)
        rag_context = rag_system.retrieve(rewritten_query, conversation_id)
        
        # Chat with chairman (using original query)
        response_dict = await chat_with_chairman(
            request.content,  # Original query
            conversation["messages"],
            rag_context,
            chairman_model=chairman_model
        )
        
        # Add simple chat message
        storage.add_chat_message(conversation_id, response_dict["content"])
        
        return {
            "type": "chat",
            "content": response_dict["content"],
            "reasoning": response_dict.get("reasoning")
        }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the response (Council or Chat).
    """
    # Check if conversation exists
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Determine mode
    is_first_message = len(conversation["messages"]) == 0
    mode = request.mode
    
    if mode == "auto":
        mode = "council" if is_first_message else "chat"

    async def event_generator():
        try:
            # Add user message
            storage.add_user_message(conversation_id, request.content)

            # Get model configuration from conversation metadata
            metadata = conversation.get("metadata", {})
            council_models = metadata.get("council_models")
            chairman_model = metadata.get("chairman_model")

            if mode == "council":
                # Start title generation in parallel (don't await yet)
                title_task = None
                if is_first_message:
                    title_task = asyncio.create_task(generate_conversation_title(request.content))

                # Stage 1: Collect responses
                yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
                stage1_results = await stage1_collect_responses(request.content, models=council_models)
                yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

                # Stage 2: Collect rankings
                yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
                stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results, models=council_models)
                aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
                yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

                # Calculate quality metrics for confidence scoring
                from .council import calculate_quality_metrics
                quality_metrics = calculate_quality_metrics(stage2_results, label_to_model)

                # Stage 3: Synthesize final answer with confidence
                yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
                stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results, label_to_model, quality_metrics, chairman_model=chairman_model)
                yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

                # Wait for title generation if it was started
                if title_task:
                    title = await title_task
                    storage.update_conversation_title(conversation_id, title)
                    yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

                # Save complete assistant message
                storage.add_assistant_message(
                    conversation_id,
                    stage1_results,
                    stage2_results,
                    stage3_result
                )

                # Calculate turn_index BEFORE using it
                updated_conversation = storage.get_conversation(conversation_id)
                turn_index = get_turn_index(updated_conversation) - 1
                
                # Index for RAG with enhanced metadata
                logger.info("[PHASE1] Indexing turn %d for conversation %s", turn_index, conversation_id)
                
                # Extract topics from question + final answer
                from .council import extract_topics
                combined_text = request.content + " " + stage3_result.get('response', '')
                topics = await extract_topics(combined_text, max_topics=3)
                logger.info("[PHASE1] Topics extracted: %s", topics)
                
                # quality_metrics already calculated on line 327, reuse it
                logger.info("[PHASE1] Quality metrics: %s", quality_metrics)
                
                # Index session with enhanced metadata
                rag_system.index_session(
                    conversation_id,
                    turn_index,
                    request.content,
                    stage1_results,
                    stage2_results,
                    stage3_result,
                    topics,
                    quality_metrics,
                )
                logger.info("[PHASE1] Session indexed successfully")
                
                # Refresh hybrid index after indexing
                rag_system.refresh_hybrid_index()
                logger.info("[PHASE1] Hybrid index refreshed")
            
            else:
                # Chat mode
                yield f"data: {json.dumps({'type': 'chat_start'})}\n\n"
                
                logger.info(f"[CHAT] Chat mode started for query: {request.content[:50]}...")
                
                # Reload conversation to get history
                updated_conversation = storage.get_conversation(conversation_id)
                logger.info(f"[CHAT] Loaded conversation with {len(updated_conversation['messages'])} messages")
                
                # PHASE 1: Rewrite query for better RAG retrieval
                from .council import rewrite_query
                logger.info(f"[CHAT] About to rewrite query...")
                rewritten_query = await rewrite_query(
                    request.content,
                    updated_conversation["messages"]
                )
                logger.info(f"[CHAT] Query rewritten, now retrieving RAG context...")
                
                # Retrieve context via RAG (using rewritten query)
                rag_context = rag_system.retrieve(rewritten_query, conversation_id)
                logger.info(f"[CHAT] RAG context retrieved ({len(rag_context)} chars), calling chairman...")
                
                # Chat with chairman (using original query)
                try:
                    logger.info(f"[CHAT] Calling chairman with query: {request.content[:50]}...")
                    response_dict = await chat_with_chairman(
                        request.content,  # Original query to Chairman
                        updated_conversation["messages"],
                        rag_context,
                        chairman_model=chairman_model
                    )
                    logger.info(f"[CHAT] Chairman response received")
                except Exception as e:
                    logger.error(f"[CHAT] Error from chairman: {e}")
                    response_dict = {
                        "content": f"I apologize, but I encountered an error: {str(e)}",
                        "usage": {}
                    }
                
                # Save chat message
                logger.info(f"[CHAT] Saving chat message...")
                storage.add_chat_message(conversation_id, response_dict["content"])
                
                yield f"data: {json.dumps({'type': 'chat_response', 'data': response_dict})}\n\n"
                logger.info(f"[CHAT] Chat response sent to client")

            # Calculate total cost for this turn
            turn_cost = 0.0
            
            if mode == "council":
                # Stage 1 costs
                for res in stage1_results:
                    turn_cost += calculate_cost(res.get('usage', {}), res['model'])
                
                # Stage 2 costs
                for res in stage2_results:
                    turn_cost += calculate_cost(res.get('usage', {}), res['model'])
                
                # Stage 3 cost
                turn_cost += calculate_cost(stage3_result.get('usage', {}), stage3_result['model'])
                
            else:
                # Chat cost
                turn_cost += calculate_cost(response_dict.get('usage', {}), chairman_model)

            # Update conversation cost
            storage.update_conversation_cost(conversation_id, turn_cost)
            
            # Get updated total cost
            updated_conv = storage.get_conversation(conversation_id)
            total_cost = updated_conv.get('total_cost', 0.0)

            # Send completion event with cost info
            yield f"data: {json.dumps({'type': 'complete', 'data': {'turn_cost': turn_cost, 'total_cost': total_cost}})}\n\n"

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )



@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, extract text (or describe image), and return the content.
    """
    result = await extract_text_from_file(file)
    
    if result["error"]:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return {
        "text": result["text"],
        "filename": file.filename,
        "truncated": result["truncated"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
