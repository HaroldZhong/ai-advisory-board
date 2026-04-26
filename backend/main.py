"""FastAPI backend for AI Advisory Board."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio

from . import config, storage
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

EXTRA_MODEL_PRICING = {
    # Web search models used by backend.web_search but not present in the curated registry.
    "perplexity/sonar": {"input": 1.0, "output": 1.0},
    "perplexity/sonar-pro": {"input": 3.0, "output": 15.0},
}

def get_turn_index(conversation: Dict[str, Any]) -> int:
    """Count the number of completed Council turns (messages with stage3)."""
    count = 0
    for msg in conversation.get("messages", []):
        if msg.get("role") == "assistant" and "stage3" in msg:
            count += 1
    return count

def calculate_cost(usage: Dict[str, int], model_id: str) -> float:
    """Calculate cost based on usage and model pricing."""
    if not usage or not model_id:
        return 0.0
    
    from .config import AVAILABLE_MODELS
    model_config = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    pricing = model_config.get('pricing', {}) if model_config else EXTRA_MODEL_PRICING.get(model_id)
    if not pricing:
        return 0.0

    input_price = pricing.get('input', 0.0)
    output_price = pricing.get('output', 0.0)
    
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    
    cost = (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price
    return cost


def calculate_turn_cost(
    mode: str,
    stage1_results: List[Dict[str, Any]] = None,
    stage2_results: List[Dict[str, Any]] = None,
    stage3_result: Dict[str, Any] = None,
    response_dict: Dict[str, Any] = None,
    chairman_model: str = None,
    extra_usage_records: List[Dict[str, Any]] = None,
) -> float:
    """Calculate authoritative backend cost for a full turn."""
    turn_cost = 0.0

    for record in extra_usage_records or []:
        turn_cost += calculate_cost(record.get("usage", {}), record.get("model"))

    if mode == "council":
        for res in stage1_results or []:
            turn_cost += calculate_cost(res.get("usage", {}), res.get("model"))
        for res in stage2_results or []:
            turn_cost += calculate_cost(res.get("usage", {}), res.get("model"))
        if stage3_result:
            turn_cost += calculate_cost(stage3_result.get("usage", {}), stage3_result.get("model"))
    else:
        turn_cost += calculate_cost((response_dict or {}).get("usage", {}), chairman_model)

    return turn_cost

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
    web_search_enabled: bool = False  # Enable Stage 0 web search
    web_search_depth: str = "fast"  # "fast" (sonar) or "deep" (sonar-pro)
    custom_instructions: str = ""  # Custom persona/instructions from user
    zdr_enabled: bool = False  # Restrict OpenRouter calls to Zero Data Retention endpoints
    execution_mode: str = "auto"  # "auto", "quick", "standard", or "research"
    rag_preset: str = "auto"  # "auto", "low", "medium", "high", or "max"
    model_tier: str = "auto"  # "auto", "budget", "mid", or "premium"
    edit_index: int = -1  # If >= 0, truncate conversation to this message index before sending


VALID_EXECUTION_MODES = {"auto", "quick", "standard", "research"}
VALID_RAG_PRESETS = {"auto", "low", "medium", "high", "max"}
VALID_MODEL_TIERS = {"auto", "budget", "mid", "premium"}


def validate_advanced_message_settings(request: SendMessageRequest) -> None:
    """Reject advanced UI settings the backend cannot honor."""
    if request.execution_mode not in VALID_EXECUTION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid execution_mode: {request.execution_mode}",
        )
    if request.rag_preset not in VALID_RAG_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rag_preset: {request.rag_preset}",
        )
    if request.model_tier not in VALID_MODEL_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_tier: {request.model_tier}",
        )


def resolve_chairman_model_for_request(
    chairman_model: Optional[str],
    request: SendMessageRequest,
) -> Optional[str]:
    """Apply model-tier override when the user explicitly set one."""
    if request.model_tier == "auto":
        return chairman_model

    from .execution_modes import select_chairman_for_tier

    return select_chairman_for_tier(request.model_tier, chairman_model)


class SessionPolicyUpdate(BaseModel):
    """Request to update a conversation's session budget policy."""
    budget_usd: Optional[float] = None
    notify_thresholds: Optional[List[float]] = None
    mode: str = "auto"
    allow_overage: bool = True


def normalize_session_policy(update: SessionPolicyUpdate) -> Dict[str, Any]:
    """Validate and normalize user-provided session policy settings."""
    update_dict = update.model_dump(exclude_unset=True)
    policy = {**config.SESSION_POLICY_DEFAULTS, **update_dict}

    budget = policy.get("budget_usd")
    if budget is not None:
        try:
            budget = float(budget)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="budget_usd must be a number or null")
        if budget <= 0:
            raise HTTPException(status_code=400, detail="budget_usd must be greater than 0")
        policy["budget_usd"] = budget

    thresholds = policy.get("notify_thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        raise HTTPException(status_code=400, detail="notify_thresholds must be a non-empty list")

    normalized_thresholds = []
    for threshold in thresholds:
        try:
            normalized_threshold = float(threshold)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="notify_thresholds must be numeric")
        if normalized_threshold <= 0:
            raise HTTPException(status_code=400, detail="notify_thresholds must be greater than 0")
        normalized_thresholds.append(normalized_threshold)
    if normalized_thresholds != sorted(normalized_thresholds):
        raise HTTPException(status_code=400, detail="notify_thresholds must be sorted ascending")
    policy["notify_thresholds"] = normalized_thresholds

    if policy.get("mode") != "auto":
        raise HTTPException(status_code=400, detail="Only auto session policy mode is supported")

    policy["allow_overage"] = bool(policy.get("allow_overage", True))
    return policy


def build_session_budget_state(conversation_id: str) -> Dict[str, Any]:
    """Return persisted policy, usage, and derived budget percentage."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "policy": storage.get_session_policy(conversation_id),
        "usage": storage.get_session_usage(conversation_id),
        "budget_spent_pct": storage.get_budget_spent_percentage(conversation_id),
    }


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


# Phase 5: Folder Management Models
class FolderCreate(BaseModel):
    name: str
    color: Optional[str] = None

class FolderUpdate(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    folder_id: Optional[str] = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "AI Advisory Board API"}


@app.get("/api/models")
async def get_models():
    """Get list of available models with live pricing from OpenRouter."""
    from .config import CURATED_MODELS
    from .openrouter_client import get_enriched_models
    
    enriched = await get_enriched_models(CURATED_MODELS)
    return {"models": enriched}


@app.get("/api/config/status")
async def get_config_status():
    """Check if the system is configured (API key exists)."""
    return {"has_api_key": config.has_openrouter_api_key()}


LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost"}


def ensure_local_setup_request(request: Request) -> None:
    """Limit setup writes to local clients in the local-first app."""
    if request.client is None:
        return

    if request.client.host not in LOCAL_CLIENT_HOSTS:
        raise HTTPException(status_code=403, detail="Setup is only available from localhost")


@app.post("/api/config/setup")
async def setup_config(data: dict, request: Request):
    """Save the OpenRouter API key to .env file."""
    ensure_local_setup_request(request)

    api_key = data.get("api_key", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    try:
        config.save_openrouter_api_key(api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    return {"success": True}



@app.get("/api/analytics")
async def get_analytics_data():
    """Get model performance analytics."""
    from .analytics import get_analytics
    return get_analytics()





@app.get("/api/folders")
async def get_folders():
    """List all folders."""
    return storage.list_folders()


@app.post("/api/folders")
async def create_folder(folder: FolderCreate):
    """Create a new folder."""
    import uuid
    folder_id = str(uuid.uuid4())
    return storage.create_folder(folder_id, folder.name, folder.color)


@app.put("/api/folders/{folder_id}")
async def update_folder(folder_id: str, updates: FolderUpdate):
    """Update a folder."""
    updated = storage.update_folder(folder_id, updates.dict(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Folder not found")
    return updated


@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: str):
    """Delete a folder."""
    if not storage.delete_folder(folder_id):
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"success": True}


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


@app.get("/api/conversations/{conversation_id}/session-policy")
async def get_session_policy_endpoint(conversation_id: str):
    """Get a conversation's persisted session budget policy and usage state."""
    return build_session_budget_state(conversation_id)


@app.put("/api/conversations/{conversation_id}/session-policy")
async def update_session_policy_endpoint(conversation_id: str, update: SessionPolicyUpdate):
    """Persist a conversation's session budget policy."""
    if storage.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    storage.set_session_policy(conversation_id, normalize_session_policy(update))
    return build_session_budget_state(conversation_id)


@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, updates: ConversationUpdate):
    """Update conversation title or folder."""
    updates_dict = updates.dict(exclude_unset=True)
    if "title" in updates_dict and updates.title is not None:
        storage.update_conversation_title(conversation_id, updates.title)
    if "folder_id" in updates_dict:
        storage.update_conversation_folder(conversation_id, updates.folder_id)
        # Keep PageIndex folder routing in sync
        rag_system.update_conversation_folder(conversation_id, updates.folder_id or "root")
        
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation entirely."""
    if not storage.delete_conversation(conversation_id):
         raise HTTPException(status_code=404, detail="Conversation not found")
    # Purge PageIndex memories for this conversation
    rag_system.delete_conversation_memories(conversation_id)
    return {"success": True}


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process OR chat with chairman.
    """
    validate_advanced_message_settings(request)

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
        title = await generate_conversation_title(
            request.content,
            zdr_enabled=request.zdr_enabled,
        )
        storage.update_conversation_title(conversation_id, title)

    # Get model configuration from conversation metadata
    metadata = conversation.get("metadata", {})
    council_models = metadata.get("council_models")
    chairman_model = resolve_chairman_model_for_request(
        metadata.get("chairman_model"),
        request,
    )

    if mode == "council":
        # Run the 3-stage council process (now with Stage 0)
        # Note: We discard steward_usage in Sync mode for now as the contract didn't ask for it in the return dict
        # But we should arguably add it to metadata if we wanted perfection. 
        # For now, we mainly care about Streaming.
        stage1_results, stage2_results, stage3_result, metadata, evidence_pack = await run_full_council(
            request.content,
            council_models=council_models,
            chairman_model=chairman_model,
            zdr_enabled=request.zdr_enabled,
        )
        extra_usage_records = []
        if metadata.get("steward_usage"):
            extra_usage_records.append({
                "model": metadata.get("steward_model") or chairman_model,
                "usage": metadata["steward_usage"],
            })
        turn_cost = calculate_turn_cost(
            mode="council",
            stage1_results=stage1_results,
            stage2_results=stage2_results,
            stage3_result=stage3_result,
            extra_usage_records=extra_usage_records,
        )

        # Add assistant message with all stages and metadata
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result,
            metadata,  # Contains label_to_model for analytics
            running_cost=turn_cost,
        )
        storage.update_conversation_cost(conversation_id, turn_cost)
        budget_state = storage.record_session_usage(conversation_id, turn_cost)

        # Calculate turn_index before logging or indexing this turn.
        updated_conversation = storage.get_conversation(conversation_id)
        turn_index = get_turn_index(updated_conversation) - 1

        # Index the session for RAG with enhanced metadata
        logger.info("[PHASE1] Indexing turn %d for conversation %s", turn_index, conversation_id)
        
        # Extract topics from question + final answer
        from .council import extract_topics, calculate_quality_metrics
        combined_text = request.content + " " + stage3_result.get('response', '')
        topics = await extract_topics(
            combined_text,
            max_topics=3,
            zdr_enabled=request.zdr_enabled,
        )
        
        # Calculate quality metrics from Stage 2 rankings
        quality_metrics = calculate_quality_metrics(
            stage2_results=stage2_results,
            label_to_model=metadata["label_to_model"],
        )
        
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
            "metadata": metadata,
            "evidence": evidence_pack.model_dump() if evidence_pack else None,
            "turn_cost": turn_cost,
            "total_cost": updated_conversation.get("total_cost", 0.0),
            "session_usage": budget_state["usage"],
            "budget_spent_pct": budget_state["budget_spent_pct"],
        }
    else:
        # Chat with Chairman
        # Reload conversation to get the latest user message we just added
        conversation = storage.get_conversation(conversation_id)
        
        # PHASE 1: Rewrite query for better RAG retrieval
        from .council import rewrite_query
        rewritten_query = await rewrite_query(
            request.content,
            conversation["messages"],
            zdr_enabled=request.zdr_enabled,
        )
        
        # Retrieve context via PageIndex reasoning RAG (using rewritten query)
        from .budget_router import create_run_plan
        run_plan = create_run_plan(
            query=request.content,
            conversation_id=conversation_id,
            has_files=bool(request.attachment_ids),
            chairman_model=chairman_model,
            execution_mode=request.execution_mode,
            rag_preset=request.rag_preset,
            model_tier=request.model_tier,
        )

        rag_context = await rag_system.retrieve_async(
            rewritten_query,
            conversation_id,
            max_tokens=run_plan.rag_max_tokens,
            zdr_enabled=request.zdr_enabled,
        )
        
        # Chat with chairman (using original query)
        response_dict = await chat_with_chairman(
            request.content,  # Original query
            conversation["messages"],
            rag_context,
            chairman_model=run_plan.chairman_model or chairman_model,
            zdr_enabled=request.zdr_enabled,
        )
        
        # Add simple chat message
        turn_cost = calculate_turn_cost(
            mode="chat",
            response_dict=response_dict,
            chairman_model=run_plan.chairman_model or chairman_model,
        )
        storage.add_chat_message(conversation_id, response_dict["content"], running_cost=turn_cost)
        storage.update_conversation_cost(conversation_id, turn_cost)
        budget_state = storage.record_session_usage(conversation_id, turn_cost)
        updated_conversation = storage.get_conversation(conversation_id)
        
        return {
            "type": "chat",
            "content": response_dict["content"],
            "reasoning": response_dict.get("reasoning"),
            "turn_cost": turn_cost,
            "total_cost": updated_conversation.get("total_cost", 0.0),
            "session_usage": budget_state["usage"],
            "budget_spent_pct": budget_state["budget_spent_pct"],
            "run_plan": run_plan.to_dict(),
        }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the response (Council or Chat).
    """
    validate_advanced_message_settings(request)

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
            current_conversation = conversation

            # Build attachment context if attachment_ids provided
            attachment_context = ""
            has_attachments = bool(request.attachment_ids)
            if has_attachments:
                attachment_context = build_llm_context(request.attachment_ids)
                logger.info(f"[ATTACH] Built context from {len(request.attachment_ids)} attachments ({len(attachment_context)} chars)")
                # Index documents into PageIndex for cross-conversation retrieval
                for att_id in request.attachment_ids:
                    att_text = get_attachment_text(att_id)
                    if att_text:
                        rag_system.index_document(conversation_id, att_id, att_text)
            
            # Combine user content with attachment context for LLM
            # User sees only their message, LLM sees message + attachments
            llm_content = request.content
            if attachment_context:
                llm_content = f"{request.content}\n\n{attachment_context}"
            
            # Prepend custom instructions as a persona prefix
            if request.custom_instructions.strip():
                llm_content = f"[User Instructions]\n{request.custom_instructions.strip()}\n\n{llm_content}"

            extra_usage_records = []
            
            # Edit & Regenerate: truncate messages if edit_index is set
            if request.edit_index >= 0:
                storage.truncate_messages(conversation_id, request.edit_index)
                # Re-fetch conversation after truncation
                current_conversation = storage.get_conversation(conversation_id)
                yield f"data: {json.dumps({'type': 'edit_truncated', 'data': {'edit_index': request.edit_index}})}\n\n"

            # Add user message (store only original content, not attachment text)
            storage.add_user_message(conversation_id, request.content)

            # Get model configuration from conversation metadata
            metadata = current_conversation.get("metadata", {})
            council_models = metadata.get("council_models")
            chairman_model = resolve_chairman_model_for_request(
                metadata.get("chairman_model"),
                request,
            )

            if mode == "council":
                # Start title generation in parallel (don't await yet)
                title_task = None
                if is_first_message:
                    title_task = asyncio.create_task(
                        generate_conversation_title(
                            request.content,
                            zdr_enabled=request.zdr_enabled,
                        )
                    )

                # Stage 0a: Web Search (if enabled)
                web_context = ""
                if request.web_search_enabled:
                    from .web_search import web_search_stage0
                    yield f"data: {json.dumps({'type': 'web_search_start'})}\n\n"
                    search_result = await web_search_stage0(
                        request.content,
                        depth=request.web_search_depth,
                        zdr_enabled=request.zdr_enabled,
                    )
                    web_context = search_result.get("context", "")
                    if search_result.get("usage"):
                        extra_usage_records.append({
                            "model": search_result.get("model"),
                            "usage": search_result.get("usage", {}),
                        })
                    yield f"data: {json.dumps({'type': 'web_search_complete', 'data': {'context': web_context[:500], 'citations': search_result.get('citations', []), 'model': search_result.get('model', '')}})}\n\n"
                    if web_context:
                        llm_content = f"[Web Search Results]\n{web_context}\n\n[User Query]\n{llm_content}"

                # Stage 0b: Tool Steward
                # We need a run_id for the tool execution
                import uuid
                run_id = str(uuid.uuid4())
                
                yield f"data: {json.dumps({'type': 'steward_start'})}\n\n"
                evidence_pack, steward_usage = await run_tool_steward_phase(
                    request.content,
                    run_id,
                    chairman_model=chairman_model,
                    zdr_enabled=request.zdr_enabled,
                )
                if steward_usage:
                    extra_usage_records.append({
                        "model": chairman_model or config.CHAIRMAN_MODEL,
                        "usage": steward_usage,
                    })
                yield f"data: {json.dumps({'type': 'steward_complete', 'data': evidence_pack.model_dump(), 'usage': steward_usage})}\n\n"

                # Stage 1: Collect responses (use llm_content with attachments)
                yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
                stage1_results = await stage1_collect_responses(
                    llm_content,
                    models=council_models,
                    evidence_pack=evidence_pack,
                    zdr_enabled=request.zdr_enabled,
                )
                yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

                # Stage 2: Collect rankings
                yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
                stage2_results, label_to_model = await stage2_collect_rankings(
                    request.content,
                    stage1_results,
                    models=council_models,
                    zdr_enabled=request.zdr_enabled,
                )
                aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
                yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

                # Calculate quality metrics for confidence scoring
                from .council import calculate_quality_metrics
                quality_metrics = calculate_quality_metrics(stage2_results, label_to_model)

                # Stage 3: Synthesize final answer with confidence
                yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
                stage3_result = await stage3_synthesize_final(
                    request.content,
                    stage1_results,
                    stage2_results,
                    label_to_model,
                    quality_metrics,
                    chairman_model=chairman_model,
                    zdr_enabled=request.zdr_enabled,
                )
                yield f"data: {json.dumps({'type': 'stage3_complete', 'data': stage3_result})}\n\n"

                # Wait for title generation if it was started
                if title_task:
                    title = await title_task
                    storage.update_conversation_title(conversation_id, title)
                    yield f"data: {json.dumps({'type': 'title_complete', 'data': {'title': title}})}\n\n"

                # Save complete assistant message with metadata for analytics
                council_metadata = {
                    "label_to_model": label_to_model,
                    "aggregate_rankings": aggregate_rankings,
                    "steward_usage": steward_usage,
                    "steward_model": chairman_model or config.CHAIRMAN_MODEL,
                }
                turn_cost = calculate_turn_cost(
                    mode="council",
                    stage1_results=stage1_results,
                    stage2_results=stage2_results,
                    stage3_result=stage3_result,
                    extra_usage_records=extra_usage_records,
                )
                storage.add_assistant_message(
                    conversation_id,
                    stage1_results,
                    stage2_results,
                    stage3_result,
                    council_metadata,  # For analytics tracking
                    running_cost=turn_cost,
                )

                # Calculate turn_index BEFORE using it
                updated_conversation = storage.get_conversation(conversation_id)
                turn_index = get_turn_index(updated_conversation) - 1
                
                # Index for RAG with enhanced metadata
                logger.info("[PHASE1] Indexing turn %d for conversation %s", turn_index, conversation_id)
                
                # Extract topics from question + final answer
                from .council import extract_topics
                combined_text = request.content + " " + stage3_result.get('response', '')
                topics = await extract_topics(
                    combined_text,
                    max_topics=3,
                    zdr_enabled=request.zdr_enabled,
                )
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
                    execution_mode=request.execution_mode,
                    rag_preset=request.rag_preset,
                    model_tier=request.model_tier,
                )
                
                # Send run plan to client for observability
                yield f"data: {json.dumps({'type': 'run_plan', 'data': run_plan.to_dict()})}\n\n"
                
                # PHASE 1: Rewrite query for better RAG retrieval
                from .council import rewrite_query
                logger.info(f"[CHAT] About to rewrite query...")
                rewritten_query = await rewrite_query(
                    request.content,
                    updated_conversation["messages"],
                    zdr_enabled=request.zdr_enabled,
                )
                logger.info(f"[CHAT] Query rewritten, now retrieving RAG context...")
                
                # Web Search grounding for chat mode (if enabled)
                chat_web_context = ""
                if request.web_search_enabled:
                    from .web_search import web_search_stage0
                    yield f"data: {json.dumps({'type': 'web_search_start'})}\n\n"
                    search_result = await web_search_stage0(
                        request.content,
                        depth=request.web_search_depth,
                        zdr_enabled=request.zdr_enabled,
                    )
                    chat_web_context = search_result.get("context", "")
                    if search_result.get("usage"):
                        extra_usage_records.append({
                            "model": search_result.get("model"),
                            "usage": search_result.get("usage", {}),
                        })
                    yield f"data: {json.dumps({'type': 'web_search_complete', 'data': {'context': chat_web_context[:500], 'citations': search_result.get('citations', []), 'model': search_result.get('model', '')}})}\n\n"

                # Retrieve context via PageIndex reasoning RAG
                rag_context = await rag_system.retrieve_async(
                    rewritten_query, 
                    conversation_id, 
                    max_tokens=run_plan.rag_max_tokens,
                    zdr_enabled=request.zdr_enabled,
                )
                logger.info(f"[CHAT] RAG context retrieved ({len(rag_context)} chars), calling chairman...")
                
                # Chat with chairman (using original query + attachment context)
                try:
                    logger.info(f"[CHAT] Calling chairman with query: {request.content[:50]}...")
                    
                    # Combine RAG context with attachment context and web search
                    combined_context = rag_context
                    if attachment_context:
                        combined_context = f"{attachment_context}\n\n{rag_context}" if rag_context else attachment_context
                    if chat_web_context:
                        combined_context = f"[Web Search Results]\n{chat_web_context}\n\n{combined_context}" if combined_context else f"[Web Search Results]\n{chat_web_context}"
                    
                    response_dict = await chat_with_chairman(
                        request.content,  # Original query to Chairman
                        updated_conversation["messages"],
                        combined_context,
                        chairman_model=run_plan.chairman_model or chairman_model,
                        zdr_enabled=request.zdr_enabled,
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
                turn_cost = calculate_turn_cost(
                    mode="chat",
                    response_dict=response_dict,
                    chairman_model=run_plan.chairman_model or chairman_model,
                    extra_usage_records=extra_usage_records,
                )
                storage.add_chat_message(conversation_id, response_dict["content"], running_cost=turn_cost)
                
                yield f"data: {json.dumps({'type': 'chat_response', 'data': response_dict})}\n\n"
                logger.info(f"[CHAT] Chat response sent to client")

            # Update conversation cost
            storage.update_conversation_cost(conversation_id, turn_cost)
            
            # Update session usage after current turn cost before checking warnings.
            budget_state = storage.record_session_usage(conversation_id, turn_cost)
            warning_level = budget_state["warning_level"]
            
            # Send budget warning if threshold crossed
            if warning_level is not None:
                warning_pct = int(warning_level * 100)
                logger.info(f"[BUDGET] Emitting warning at {warning_pct}% for conversation {conversation_id}")
                yield f"data: {json.dumps({'type': 'budget_warning', 'data': {'threshold': warning_level, 'percentage': warning_pct}})}\n\n"
            
            # Get updated total cost
            updated_conv = storage.get_conversation(conversation_id)
            total_cost = updated_conv.get('total_cost', 0.0)
            
            # Get budget spent percentage for completion event
            spent_pct = budget_state["budget_spent_pct"]

            # Send completion event with cost info and budget status
            yield f"data: {json.dumps({'type': 'complete', 'data': {'turn_cost': turn_cost, 'total_cost': total_cost, 'session_usage': budget_state['usage'], 'budget_spent_pct': spent_pct}})}\n\n"

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
async def upload_file(
    file: UploadFile = File(...),
    use_zdr: bool = False,
):
    """
    Legacy: Upload a file, extract text (or describe image), and return the content.
    Use /api/attachments for new implementation.
    """
    result = await extract_text_from_file(file, zdr_enabled=use_zdr)
    
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
async def create_attachment_endpoint(
    file: UploadFile = File(...),
    use_zdr: bool = False,
):
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
    result = await process_file(
        content,
        file.filename,
        mime_type,
        zdr_enabled=use_zdr,
    )
    
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


# =============================================================================
# Serve Frontend Static Files
# =============================================================================
import sys
def get_base_path():
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return sys._MEIPASS
    except Exception:
        return os.path.dirname(os.path.dirname(__file__))

frontend_dir = os.path.join(get_base_path(), "frontend", "dist")

if os.path.exists(frontend_dir):
    assets_dir = os.path.join(frontend_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
        
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Ignore API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
            
        # Try to serve a specific file if it exists (e.g., favicon.ico)
        file_path = os.path.join(frontend_dir, full_path)
        if full_path and os.path.isfile(file_path):
            from fastapi.responses import FileResponse
            return FileResponse(file_path)
            
        # Otherwise, serve index.html for React Router
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return {"error": "Frontend build not found"}
else:
    logger.warning("Frontend build directory not found. Please run 'npm run build' in the frontend folder.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
