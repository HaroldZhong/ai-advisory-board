"""FastAPI backend for AI Advisory Board."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import json
import asyncio

from . import storage
from .council import run_full_council, generate_conversation_title, stage1_collect_responses, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings, chat_with_chairman, run_tool_steward_phase
from .rag import CouncilRAG
from .file_processing import extract_text_from_file, process_file, get_mime_type
from .attachment_storage import (
    create_attachment, get_attachment, update_attachment_status,
    save_attachment_text, get_attachment_text, build_llm_context,
    Attachment
)
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

app = FastAPI(title="AI Advisory Board API")

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
    attachment_ids: List[str] = []  # List of attachment IDs to include


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
    return {"status": "ok", "service": "AI Advisory Board API"}


@app.get("/api/models")
async def get_models():
    """Get list of available models with live pricing from OpenRouter."""
    from .config import CURATED_MODELS
    from .openrouter_client import get_enriched_models
    
    enriched = await get_enriched_models(CURATED_MODELS)
    return {"models": enriched}


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
        # Run the 3-stage council process (now with Stage 0)
        # Note: We discard steward_usage in Sync mode for now as the contract didn't ask for it in the return dict
        # But we should arguably add it to metadata if we wanted perfection. 
        # For now, we mainly care about Streaming.
        stage1_results, stage2_results, stage3_result, metadata, evidence_pack = await run_full_council(
            request.content,
            council_models=council_models,
            chairman_model=chairman_model
        )

        # Add assistant message with all stages and metadata
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result,
            metadata  # Contains label_to_model for analytics
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
            "stage2": stage2_results,
            "stage3": stage3_result,
            "metadata": metadata,
            "evidence": evidence_pack.dict() if evidence_pack else None
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
            # Build attachment context if attachment_ids provided
            attachment_context = ""
            has_attachments = bool(request.attachment_ids)
            if has_attachments:
                attachment_context = build_llm_context(request.attachment_ids)
                logger.info(f"[ATTACH] Built context from {len(request.attachment_ids)} attachments ({len(attachment_context)} chars)")
            
            # Combine user content with attachment context for LLM
            # User sees only their message, LLM sees message + attachments
            llm_content = request.content
            if attachment_context:
                llm_content = f"{request.content}\n\n{attachment_context}"
            
            # Add user message (store only original content, not attachment text)
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

                # Stage 0: Tool Steward
                # We need a run_id for the tool execution
                import uuid
                run_id = str(uuid.uuid4())
                
                yield f"data: {json.dumps({'type': 'steward_start'})}\n\n"
                evidence_pack, steward_usage = await run_tool_steward_phase(request.content, run_id, chairman_model=chairman_model)
                yield f"data: {json.dumps({'type': 'steward_complete', 'data': evidence_pack.dict(), 'usage': steward_usage})}\n\n"

                # Stage 1: Collect responses (use llm_content with attachments)
                yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
                stage1_results = await stage1_collect_responses(llm_content, models=council_models, evidence_pack=evidence_pack)
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

                # Save complete assistant message with metadata for analytics
                council_metadata = {
                    "label_to_model": label_to_model,
                    "aggregate_rankings": aggregate_rankings
                }
                storage.add_assistant_message(
                    conversation_id,
                    stage1_results,
                    stage2_results,
                    stage3_result,
                    council_metadata  # For analytics tracking
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
                
                # PHASE 2: Create Run Plan for budget-aware routing
                from .budget_router import create_run_plan
                run_plan = create_run_plan(
                    query=request.content,
                    conversation_id=conversation_id,
                    has_files=has_attachments,
                    chairman_model=chairman_model,
                )
                
                # Send run plan to client for observability
                yield f"data: {json.dumps({'type': 'run_plan', 'data': run_plan.to_dict()})}\n\n"
                
                # PHASE 1: Rewrite query for better RAG retrieval
                from .council import rewrite_query
                logger.info(f"[CHAT] About to rewrite query...")
                rewritten_query = await rewrite_query(
                    request.content,
                    updated_conversation["messages"]
                )
                logger.info(f"[CHAT] Query rewritten, now retrieving RAG context...")
                
                # Retrieve context via RAG using budget from Run Plan
                rag_context = rag_system.retrieve(
                    rewritten_query, 
                    conversation_id, 
                    max_tokens=run_plan.rag_max_tokens
                )
                logger.info(f"[CHAT] RAG context retrieved ({len(rag_context)} chars), calling chairman...")
                
                # Chat with chairman (using original query + attachment context)
                try:
                    logger.info(f"[CHAT] Calling chairman with query: {request.content[:50]}...")
                    
                    # Combine RAG context with attachment context
                    combined_context = rag_context
                    if attachment_context:
                        combined_context = f"{attachment_context}\n\n{rag_context}" if rag_context else attachment_context
                    
                    response_dict = await chat_with_chairman(
                        request.content,  # Original query to Chairman
                        updated_conversation["messages"],
                        combined_context,
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
            
            # Update session usage for budget tracking
            warning_level = storage.check_budget_warning(conversation_id)
            storage.update_session_usage(conversation_id, turn_cost, emit_warning=warning_level)
            
            # Send budget warning if threshold crossed
            if warning_level is not None:
                warning_pct = int(warning_level * 100)
                logger.info(f"[BUDGET] Emitting warning at {warning_pct}% for conversation {conversation_id}")
                yield f"data: {json.dumps({'type': 'budget_warning', 'data': {'threshold': warning_level, 'percentage': warning_pct}})}\n\n"
            
            # Get updated total cost
            updated_conv = storage.get_conversation(conversation_id)
            total_cost = updated_conv.get('total_cost', 0.0)
            
            # Get budget spent percentage for completion event
            spent_pct = storage.get_budget_spent_percentage(conversation_id)

            # Send completion event with cost info and budget status
            yield f"data: {json.dumps({'type': 'complete', 'data': {'turn_cost': turn_cost, 'total_cost': total_cost, 'budget_spent_pct': spent_pct}})}\n\n"

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
    Legacy: Upload a file, extract text (or describe image), and return the content.
    Use /api/attachments for new implementation.
    """
    result = await extract_text_from_file(file)
    
    if result["error"]:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return {
        "text": result["text"],
        "filename": file.filename,
        "truncated": result["truncated"]
    }


# =============================================================================
# ATTACHMENT API (New unified file upload system)
# =============================================================================

@app.post("/api/attachments")
async def create_attachment_endpoint(file: UploadFile = File(...)):
    """
    Upload a file and create an attachment.
    Returns attachment_id and status. Extraction happens async.
    """
    content = await file.read()
    mime_type = get_mime_type(file.filename, file.content_type)
    
    # Create attachment record (stores raw file)
    attachment = create_attachment(content, file.filename, mime_type)
    
    # Check if this was a cache hit (already processed)
    if attachment.status in ("success", "partial"):
        return {
            "attachment_id": attachment.attachment_id,
            "status": attachment.status,
            "filename": attachment.filename,
            "cached": True,
            "warning": attachment.warning
        }
    
    # Process the file (extraction)
    result = await process_file(content, file.filename, mime_type)
    
    # Update attachment with extraction result
    update_attachment_status(
        attachment.attachment_id,
        status=result.status,
        method=result.method,
        warning=result.warning,
        error=result.error,
        stats=result.stats
    )
    
    # Save extracted text
    if result.text:
        save_attachment_text(attachment.attachment_id, result.text)
    
    logger.info(f"[ATTACH] Processed {file.filename} -> {result.status}")
    
    return {
        "attachment_id": attachment.attachment_id,
        "status": result.status,
        "filename": file.filename,
        "cached": False,
        "method": result.method,
        "warning": result.warning,
        "error": result.error,
        "stats": result.stats
    }


@app.get("/api/attachments/{attachment_id}")
async def get_attachment_endpoint(attachment_id: str):
    """
    Get attachment metadata and status.
    """
    attachment = get_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    return attachment.model_dump()


@app.get("/api/attachments/{attachment_id}/text")
async def get_attachment_text_endpoint(attachment_id: str, preview: bool = False):
    """
    Get extracted text for an attachment.
    If preview=True, returns first 1000 characters only.
    """
    attachment = get_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    text = get_attachment_text(attachment_id)
    if text is None:
        raise HTTPException(status_code=404, detail="No text available for this attachment")
    
    if preview and len(text) > 1000:
        text = text[:1000] + "\n[...preview truncated...]"
    
    return {"text": text, "preview": preview}


@app.post("/api/attachments/{attachment_id}/enhance")
async def enhance_attachment_endpoint(
    attachment_id: str,
    engine: str = "pdf-text",
    use_zdr: bool = False
):
    """
    Re-extract attachment content using OpenRouter enhanced PDF processing.
    
    Use this when local extraction failed or produced poor results.
    
    Args:
        engine: "pdf-text" (free) or "mistral-ocr" (paid, better for scans)
        use_zdr: Enable Zero Data Retention for privacy
    """
    from .openrouter_pdf import extract_pdf_with_openrouter, estimate_pdf_cost
    from .attachment_storage import get_attachment_raw
    
    attachment = get_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Only PDFs can be enhanced via OpenRouter
    if attachment.mime_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="Enhanced extraction only available for PDF files"
        )
    
    # Get raw file content
    content = get_attachment_raw(attachment_id)
    if not content:
        raise HTTPException(status_code=404, detail="Raw file not found")
    
    logger.info(f"[ATTACH] Enhancing {attachment_id} with engine={engine}")
    
    # Process with OpenRouter
    result = await extract_pdf_with_openrouter(
        content,
        attachment.filename,
        engine=engine,
        use_zdr=use_zdr
    )
    
    # Update attachment with new extraction
    if result["status"] == "success":
        method = f"openrouter_{engine.replace('-', '_')}"
        update_attachment_status(
            attachment_id,
            status="success",
            method=method,
            warning=None,
            error=None,
            stats={
                "char_count": len(result["text"]),
                "page_count": attachment.stats.page_count
            }
        )
        save_attachment_text(attachment_id, result["text"])
    else:
        update_attachment_status(
            attachment_id,
            status=result["status"],
            method=f"openrouter_{engine.replace('-', '_')}",
            warning=result.get("error"),
        )
    
    return {
        "attachment_id": attachment_id,
        "status": result["status"],
        "method": f"openrouter_{engine.replace('-', '_')}",
        "char_count": len(result.get("text", "")),
        "cost": result.get("cost", 0.0),
        "error": result.get("error"),
    }


@app.get("/api/attachments/{attachment_id}/recommendation")
async def get_extraction_recommendation(attachment_id: str):
    """
    Get recommendation for enhanced extraction based on local extraction quality.
    
    Returns recommendation on whether enhanced extraction would help and which engine to use.
    """
    from .openrouter_pdf import get_engine_recommendation
    
    attachment = get_attachment(attachment_id)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Only PDFs can be enhanced
    if attachment.mime_type != "application/pdf":
        return {
            "needs_enhanced": False,
            "reason": "Enhanced extraction only available for PDFs",
        }
    
    # Get recommendation based on stats
    recommendation = get_engine_recommendation(
        char_count=attachment.stats.char_count,
        empty_page_ratio=attachment.stats.empty_page_ratio,
        page_count=attachment.stats.page_count or 1
    )
    
    return recommendation


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
