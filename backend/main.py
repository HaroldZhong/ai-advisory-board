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

from . import app_paths, config, storage
from .council import generate_conversation_title, stage1_collect_responses, stage1_collect_responses_progressive, stage2_collect_rankings, stage3_synthesize_final, calculate_aggregate_rankings, chat_with_chairman, run_tool_steward_phase
from .turn_pipeline import run_turn
from .conversation_export import (
    build_conversation_markdown,
    get_conversation_export_filename,
    resolve_unique_export_path,
)
from .rag import CouncilRAG
from .file_processing import process_file, get_mime_type
from .attachment_storage import (
    create_attachment, get_attachment, update_attachment_status,
    save_attachment_text, get_attachment_text, build_llm_context,
    link_attachments_to_conversation, delete_attachment,
    delete_attachments_for_conversation, collect_attachment_ids_from_messages,
    Attachment
)
from .logger import logger
from .reasoning_stream import ReasoningStreamState

# Initialize RAG system
rag_system = CouncilRAG()

def calculate_cost(usage: Dict[str, int], model_id: str) -> float:
    """Calculate cost based on usage and model pricing."""
    if not usage or not model_id:
        return 0.0
    
    from .config import AVAILABLE_MODELS
    model_config = next((m for m in AVAILABLE_MODELS if m['id'] == model_id), None)
    pricing = model_config.get('pricing', {}) if model_config else None
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


def build_reasoning_stream_events(
    response: Dict[str, Any],
    *,
    scope: str,
    stage: str,
    model: Optional[str],
    content_key: str,
    index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize completed model response text/reasoning into SSE event payloads."""
    delta: Dict[str, Any] = {}
    reasoning = response.get("reasoning") or response.get("reasoning_details")
    if reasoning:
        delta["reasoning_details"] = reasoning

    content = response.get(content_key)
    if isinstance(content, str) and content:
        delta["content"] = content

    if not delta:
        return []

    state = ReasoningStreamState("openrouter_unified")
    events = []
    for event in state.consume_delta(delta):
        event_data = {
            "scope": scope,
            "stage": stage,
            "model": model,
            "text": event["text"],
        }
        if index is not None:
            event_data["index"] = index
        for key in ("detail_type", "format"):
            if event.get(key) is not None:
                event_data[key] = event[key]
        events.append({"type": event["type"], "data": event_data})
    return events


def encode_sse_event(event: Dict[str, Any]) -> str:
    """Encode a normalized event dict as a server-sent event line."""
    return f"data: {json.dumps(event)}\n\n"

app = FastAPI(title="AI Advisory Board API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


VALID_DEFAULT_MODES = {"chat", "council"}


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    topic: str = "New Conversation"
    council_members: Optional[List[str]] = None
    chairman_model: Optional[str] = None
    preset_id: Optional[str] = None
    zdr_enabled: Optional[bool] = None
    budget_usd: Optional[float] = None
    budget_allow_overage: bool = False
    thinking_effort: Optional[str] = None
    default_mode: Optional[str] = None


@app.post("/api/conversations")
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    metadata = {}
    preset = None
    session_policy = None

    if request.default_mode is not None:
        if request.default_mode not in VALID_DEFAULT_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid default_mode: {request.default_mode}",
            )
        metadata["default_mode"] = request.default_mode

    if request.preset_id:
        preset = next(
            (candidate for candidate in config.MODEL_PRESETS if candidate["id"] == request.preset_id),
            None,
        )
        if preset is None:
            raise HTTPException(status_code=400, detail=f"Invalid model preset: {request.preset_id}")
        if preset.get("requires_zdr") and request.zdr_enabled is False:
            raise HTTPException(status_code=400, detail=f"Preset {request.preset_id} requires ZDR")
        metadata["preset_id"] = request.preset_id

    council_members = request.council_members or None
    if council_members is None and preset is not None:
        council_members = preset["council_models"]

    chairman_model = (request.chairman_model or "").strip() or None
    if chairman_model is None and preset is not None:
        chairman_model = preset["chairman_model"]

    # Effective ZDR: explicit request value, else implied by a requires_zdr
    # preset. Must gate on this — not just the explicit flag — or a preset
    # like "private" silently loses its ZDR guarantee off-OpenRouter.
    effective_zdr = (
        bool(request.zdr_enabled)
        if request.zdr_enabled is not None
        else bool(preset is not None and preset.get("requires_zdr"))
    )
    if effective_zdr and not config.provider_is_openrouter():
        raise HTTPException(status_code=400, detail="ZDR requires OpenRouter")

    if request.zdr_enabled is not None:
        metadata["zdr_enabled"] = bool(request.zdr_enabled)
    elif preset is not None and preset.get("requires_zdr"):
        metadata["zdr_enabled"] = True

    if request.thinking_effort is not None:
        metadata["thinking_effort"] = cap_thinking_effort_for_preset(
            request.preset_id,
            validate_thinking_effort(request.thinking_effort),
        )
    elif preset is not None:
        metadata["thinking_effort"] = cap_thinking_effort_for_preset(
            request.preset_id,
            preset.get("default_reasoning_effort", "medium"),
        )

    if request.budget_usd is not None:
        session_policy = normalize_session_policy(SessionPolicyUpdate(
            budget_usd=request.budget_usd,
            allow_overage=request.budget_allow_overage,
        ))

    models_by_id = {m['id']: m for m in config.AVAILABLE_MODELS}
    # Off-OpenRouter, a provider (local server, relay) can serve models the
    # curated registry has never heard of — accept any non-empty id it isn't
    # in the registry for. Registry HITS still go through the existing
    # type checks below on every provider kind, and empty/whitespace ids are
    # rejected everywhere via the `m not in models_by_id` / falsy checks.
    allow_unregistered = not config.provider_is_openrouter()

    # Validate council members
    if council_members:
        invalid = [
            m for m in council_members
            if m not in models_by_id and not (allow_unregistered and m.strip())
        ]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid council models: {invalid}"
            )
        # utility-type models (e.g. the RAG extraction model) and search-type
        # models (e.g. perplexity/sonar) exist only for internal cost
        # accounting and dedicated web-search calls, and are never
        # user-selectable as chairman or council.
        utility = [m for m in council_members if models_by_id.get(m, {}).get("type") == "utility"]
        if utility:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid council models (internal utility model, not selectable): {utility}",
            )
        search = [m for m in council_members if models_by_id.get(m, {}).get("type") == "search"]
        if search:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid council models (internal search model, not selectable): {search}",
            )
        metadata["council_models"] = council_members

    # Validate chairman model
    if chairman_model:
        known_chairman = models_by_id.get(chairman_model)
        if known_chairman is None and not (allow_unregistered and chairman_model.strip()):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chairman model: {chairman_model}"
            )
        if known_chairman is not None and known_chairman.get("type") == "utility":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chairman model (internal utility model, not selectable): {chairman_model}",
            )
        if known_chairman is not None and known_chairman.get("type") == "search":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chairman model (internal search model, not selectable): {chairman_model}",
            )
        metadata["chairman_model"] = chairman_model

    if metadata.get("zdr_enabled") is True:
        ensure_zdr_compatible_models(chairman_model, council_members)
        
    conversation = storage.create_conversation(conversation_id, metadata)
    if session_policy is not None:
        storage.set_session_policy(conversation_id, session_policy)
        conversation = storage.get_conversation(conversation_id) or conversation
    return conversation


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    content: str
    mode: str = "auto"  # "auto", "council", or "chat"
    attachment_ids: List[str] = []  # List of attachment IDs to include
    web_search_enabled: bool = False  # Enable Stage 0 web search
    web_search_depth: str = "fast"  # "fast" (sonar) or "deep" (sonar-pro)
    custom_instructions: str = ""  # Custom persona/instructions from user
    zdr_enabled: Optional[bool] = None  # Restrict OpenRouter calls to Zero Data Retention endpoints
    execution_mode: str = "auto"  # "auto", "quick", "standard", or "research"
    rag_preset: str = "auto"  # "auto", "low", "medium", "high", or "max"
    model_tier: str = "auto"  # "auto", "budget", "mid", or "premium"
    thinking_effort: Optional[str] = None  # "minimal", "low", "medium", "high", or "xhigh"
    edit_index: int = -1  # If >= 0, truncate conversation to this message index before sending


VALID_EXECUTION_MODES = {"auto", "quick", "standard", "research"}
VALID_RAG_PRESETS = {"auto", "low", "medium", "high", "max"}
VALID_MODEL_TIERS = {"auto", "budget", "mid", "premium"}
VALID_THINKING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
THINKING_EFFORT_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
THINKING_EFFORT_MAX_BY_PRESET = {"budget": "medium"}


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Return curated model metadata for a registry id."""
    return next((model for model in config.CURATED_MODELS if model["id"] == model_id), None)


def ensure_zdr_compatible_models(
    chairman_model: Optional[str],
    council_models: Optional[List[str]],
) -> None:
    """Reject model selections that cannot satisfy conversation-level ZDR."""
    selected_chairman = chairman_model or config.CHAIRMAN_MODEL
    selected_council = council_models or config.COUNCIL_MODELS
    incompatible = []

    for model_id in [selected_chairman, *selected_council]:
        model = get_model_by_id(model_id)
        if model is None or model.get("supports_zdr") is not True:
            incompatible.append(model_id)

    if incompatible:
        raise HTTPException(
            status_code=400,
            detail=f"ZDR-enabled conversations require ZDR-capable models: {incompatible}",
        )


def resolve_effective_zdr(conversation: Dict[str, Any], request: SendMessageRequest) -> bool:
    """Resolve ZDR for a turn. Stored conversation privacy cannot be downgraded per message."""
    metadata = conversation.get("metadata", {})
    # Privacy-sticky: once a conversation stores ZDR=True, individual sends cannot
    # downgrade it. A future metadata update endpoint must make that change explicit.
    return metadata.get("zdr_enabled") is True or request.zdr_enabled is True


def validate_thinking_effort(thinking_effort: str) -> str:
    """Validate a user-facing thinking effort value."""
    if thinking_effort not in VALID_THINKING_EFFORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid thinking_effort: {thinking_effort}",
        )
    return thinking_effort


def cap_thinking_effort_for_preset(preset_id: Optional[str], thinking_effort: str) -> str:
    """Apply preset-specific max effort caps while preserving lower user choices."""
    max_effort = THINKING_EFFORT_MAX_BY_PRESET.get(preset_id or "")
    if max_effort is None:
        return thinking_effort
    if THINKING_EFFORT_ORDER[thinking_effort] > THINKING_EFFORT_ORDER[max_effort]:
        return max_effort
    return thinking_effort


def resolve_effective_thinking_effort(
    conversation: Dict[str, Any],
    request: SendMessageRequest,
) -> str:
    """Resolve thinking effort for a turn: request > metadata > preset default > medium."""
    metadata = conversation.get("metadata", {})
    preset_id = metadata.get("preset_id")

    if request.thinking_effort is not None:
        return cap_thinking_effort_for_preset(
            preset_id,
            validate_thinking_effort(request.thinking_effort),
        )

    stored_effort = metadata.get("thinking_effort")
    if stored_effort in VALID_THINKING_EFFORTS:
        return cap_thinking_effort_for_preset(preset_id, stored_effort)

    preset = next(
        (candidate for candidate in config.MODEL_PRESETS if candidate["id"] == preset_id),
        None,
    )
    if preset is not None and preset.get("default_reasoning_effort") in VALID_THINKING_EFFORTS:
        return cap_thinking_effort_for_preset(preset_id, preset["default_reasoning_effort"])

    return "medium"


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
    if request.thinking_effort is not None:
        validate_thinking_effort(request.thinking_effort)


def validate_advanced_settings_for_mode(mode: str, request: SendMessageRequest) -> None:
    """Reject chat-routing controls for council mode instead of silently ignoring them."""
    if mode != "council":
        return

    unsupported = []
    if request.execution_mode != "auto":
        unsupported.append("execution_mode")
    if request.rag_preset != "auto":
        unsupported.append("rag_preset")

    if unsupported:
        fields = ", ".join(unsupported)
        raise HTTPException(
            status_code=400,
            detail=f"{fields} only apply to chat mode; council mode always runs the full council pipeline.",
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


def prepare_message_attachments(
    conversation_id: str,
    attachment_ids: List[str],
) -> List[Dict[str, Any]]:
    """Persist attachment references and return metadata for the user message."""
    if not attachment_ids:
        return []

    attachments = link_attachments_to_conversation(attachment_ids, conversation_id)
    return [attachment.model_dump() for attachment in attachments]


def delete_truncated_message_attachments(
    conversation_id: str,
    keep_count: int,
    keep_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Release attachments referenced only by messages removed during edit/regenerate.

    keep_ids: attachment ids being resent with the edited message (Codex
    review, P3-T8 round 2 item 1). Without this, an edit/regenerate that
    resends an attachment id only ever referenced by the truncated tail
    raced with prepare_message_attachments: this function deleted the
    attachment's files, then the resend tried to relink an id whose files
    were already gone. Skip deleting those ids here — they're about to be
    relinked to the SAME conversation, not actually removed from it.
    """
    conversation = storage.get_conversation(conversation_id)
    if not conversation:
        return {"attachment_ids": [], "deleted": 0, "retained": 0, "missing": 0, "files_deleted": 0, "results": []}

    removed_messages = conversation.get("messages", [])[keep_count:]
    attachment_ids = [
        attachment_id for attachment_id in collect_attachment_ids_from_messages(removed_messages)
        if not keep_ids or attachment_id not in keep_ids
    ]
    results = [
        delete_attachment(attachment_id, conversation_id=conversation_id)
        for attachment_id in attachment_ids
    ]

    return {
        "attachment_ids": attachment_ids,
        "deleted": sum(1 for result in results if result["deleted"]),
        "retained": sum(1 for result in results if result["retained"]),
        "missing": sum(1 for result in results if not result["found"]),
        "files_deleted": sum(result["files_deleted"] for result in results),
        "results": results,
    }


class SessionPolicyUpdate(BaseModel):
    """Request to update a conversation's session budget policy."""
    budget_usd: Optional[float] = None
    notify_thresholds: Optional[List[float]] = None
    mode: str = "auto"
    allow_overage: bool = True


def normalize_session_policy(
    update: SessionPolicyUpdate,
    base_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and normalize user-provided session policy settings."""
    update_dict = update.model_dump(exclude_unset=True)
    policy = {**config.SESSION_POLICY_DEFAULTS, **(base_policy or {}), **update_dict}

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


BUDGET_CAP_REACHED_DETAIL = "Session budget reached. Raise the cap before sending another message."


def ensure_budget_allows_new_turn(
    conversation_id: str,
    conversation: Optional[Dict[str, Any]] = None,
) -> None:
    """Reject new paid turns when an enforced session budget is already exhausted."""
    if conversation is None:
        policy = storage.get_session_policy(conversation_id)
        spent_pct = storage.get_budget_spent_percentage(conversation_id)
    else:
        policy = {**config.SESSION_POLICY_DEFAULTS, **conversation.get("session_policy", {})}
        usage = conversation.get("session_usage", {"spent_usd": 0.0})
        budget = policy.get("budget_usd")
        spent_pct = None
        if budget is not None and budget > 0:
            spent_pct = usage.get("spent_usd", 0.0) / budget

    if policy.get("allow_overage", True):
        return

    if spent_pct is not None and spent_pct >= 1.0:
        raise HTTPException(status_code=409, detail=BUDGET_CAP_REACHED_DETAIL)


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
    zdr_enabled: Optional[bool] = None
    thinking_effort: Optional[str] = None


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "AI Advisory Board API"}


@app.get("/api/models")
async def get_models():
    """Get list of available models with live pricing from OpenRouter."""
    from .config import CHAIRMAN_MODEL, COUNCIL_MODELS, CURATED_MODELS, MODEL_PRESETS
    from .openrouter_client import get_enriched_models
    
    enriched = await get_enriched_models(CURATED_MODELS)
    return {
        "models": enriched,
        "defaults": {
            "chairman": CHAIRMAN_MODEL,
            "council": COUNCIL_MODELS,
        },
        "presets": MODEL_PRESETS,
    }


@app.get("/api/config/status")
async def get_config_status():
    """Check if the system is configured (API key exists)."""
    return {
        "has_api_key": config.has_openrouter_api_key(),
        "provider_kind": config.PROVIDER_KIND,
    }


@app.get("/api/config/connectivity")
async def get_connectivity_status():
    """Probe OpenRouter reachability so the UI can distinguish network blocks from key problems."""
    from .openrouter_client import check_connectivity
    return await check_connectivity()


LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost"}


def ensure_local_setup_request(request: Request) -> None:
    """Limit setup writes to local clients in the local-first app."""
    if request.client is None:
        return

    if request.client.host not in LOCAL_CLIENT_HOSTS:
        raise HTTPException(status_code=403, detail="Setup is only available from localhost")


class ConnectivityCheckRequest(BaseModel):
    """Optional candidate key to validate before it's saved (first-run 'Test connection')."""
    api_key: Optional[str] = None


@app.post("/api/config/connectivity")
async def post_connectivity_status(data: ConnectivityCheckRequest, request: Request):
    """Same probe as GET, but validates a caller-supplied key instead of the saved one.

    Carries a secret in the body, so it gets the same localhost guard as /api/config/setup.
    """
    ensure_local_setup_request(request)
    from .openrouter_client import check_connectivity
    return await check_connectivity(api_key=data.api_key or None)


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
    updated = storage.update_folder(folder_id, updates.model_dump(exclude_unset=True))
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


@app.post("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str):
    """Export a saved conversation to Markdown in the app data exports folder."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    filename = get_conversation_export_filename(conversation.get("title"), conversation_id)
    export_path = resolve_unique_export_path(app_paths.get_exports_dir(), filename)
    markdown = build_conversation_markdown(conversation)

    try:
        app_paths.write_text_atomic(export_path, markdown)
    except (OSError, RuntimeError, UnicodeError) as exc:
        logger.error("Failed to export conversation %s to %s: %s", conversation_id, export_path, exc)
        raise HTTPException(status_code=500, detail="Failed to export conversation") from exc

    logger.info("Exported conversation %s to %s", conversation_id, export_path)
    return {"filename": export_path.name, "path": str(export_path)}


@app.get("/api/conversations/{conversation_id}/session-policy")
async def get_session_policy_endpoint(conversation_id: str):
    """Get a conversation's persisted session budget policy and usage state."""
    return build_session_budget_state(conversation_id)


@app.put("/api/conversations/{conversation_id}/session-policy")
async def update_session_policy_endpoint(conversation_id: str, update: SessionPolicyUpdate):
    """Persist a conversation's session budget policy."""
    if storage.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    existing_policy = storage.get_session_policy(conversation_id)
    storage.set_session_policy(conversation_id, normalize_session_policy(update, existing_policy))
    return build_session_budget_state(conversation_id)


@app.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, updates: ConversationUpdate):
    """Update conversation title, folder, or conversation-level privacy."""
    updates_dict = updates.model_dump(exclude_unset=True)
    conv = storage.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if "zdr_enabled" in updates_dict and updates.zdr_enabled is not None:
        metadata = conv.get("metadata", {})
        preset_id = metadata.get("preset_id")
        preset = next(
            (candidate for candidate in config.MODEL_PRESETS if candidate["id"] == preset_id),
            None,
        )
        if updates.zdr_enabled is False and preset is not None and preset.get("requires_zdr"):
            raise HTTPException(status_code=400, detail=f"Preset {preset_id} requires ZDR")
        if updates.zdr_enabled is True and not config.provider_is_openrouter():
            raise HTTPException(status_code=400, detail="ZDR requires OpenRouter")
        if updates.zdr_enabled is True:
            ensure_zdr_compatible_models(
                metadata.get("chairman_model"),
                metadata.get("council_models"),
            )
    if "thinking_effort" in updates_dict and updates.thinking_effort is not None:
        validate_thinking_effort(updates.thinking_effort)

    if "title" in updates_dict and updates.title is not None:
        storage.update_conversation_title(conversation_id, updates.title)
    if "folder_id" in updates_dict:
        storage.update_conversation_folder(conversation_id, updates.folder_id)
        # Keep PageIndex folder routing in sync
        rag_system.update_conversation_folder(conversation_id, updates.folder_id or "root")

    if "zdr_enabled" in updates_dict and updates.zdr_enabled is not None:
        storage.update_conversation_metadata(conversation_id, {"zdr_enabled": bool(updates.zdr_enabled)})
        if updates.zdr_enabled is True:
            # The startup cleanup sweep (CouncilRAG.cleanup_zdr_conversations)
            # only runs once per process, so a conversation flipping ZDR on at
            # runtime would otherwise leave its already-indexed memories live
            # and retrievable from other conversations until restart. Purge
            # immediately using the same path conversation deletion uses.
            await rag_system.delete_conversation_memories(conversation_id)
    if "thinking_effort" in updates_dict and updates.thinking_effort is not None:
        metadata = conv.get("metadata", {})
        storage.update_conversation_metadata(
            conversation_id,
            {
                "thinking_effort": cap_thinking_effort_for_preset(
                    metadata.get("preset_id"),
                    updates.thinking_effort,
                )
            },
        )

    conv = storage.get_conversation(conversation_id)
    return conv


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation entirely."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not storage.delete_conversation(conversation_id):
         raise HTTPException(status_code=404, detail="Conversation not found")
    attachment_cleanup = delete_attachments_for_conversation(conversation_id, conversation)
    # Purge PageIndex memories for this conversation
    await rag_system.delete_conversation_memories(conversation_id)
    return {"success": True, "attachments": attachment_cleanup}


def prepare_turn(conversation_id: str, request: SendMessageRequest):
    """Shared pre-flight for both message endpoints.

    Raises HTTP errors (validation, 404, budget) BEFORE the pipeline starts so
    they surface as proper status codes, not mid-stream error events.
    """
    validate_advanced_message_settings(request)

    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    zdr_enabled = resolve_effective_zdr(conversation, request)
    if zdr_enabled and not config.provider_is_openrouter():
        raise HTTPException(
            status_code=400,
            detail="ZDR requires OpenRouter. Disable ZDR for this conversation or switch providers.",
        )
    thinking_effort = resolve_effective_thinking_effort(conversation, request)

    # Determine mode. Auto-resolution must use the EFFECTIVE message count —
    # a pending edit_index truncation makes an edit-back-to-message-0 send
    # effectively first, so it routes to council like the original send did.
    is_first_message = len(conversation["messages"]) == 0
    # Clamp: truncate_messages keeps messages[:edit_index], so a stale
    # edit_index beyond the stored count leaves min(edit_index, len) messages.
    effective_message_count = (
        min(request.edit_index, len(conversation["messages"]))
        if request.edit_index >= 0
        else len(conversation["messages"])
    )
    # Mode resolution order (P3-T3, master plan P3-W2, owner decision #2):
    # 1. An explicit request.mode ("chat"/"council") always wins.
    # 2. metadata.default_mode == "chat": every turn runs chat, including the first.
    # 3. metadata.default_mode == "council": council on the effectively-first
    #    turn, chat after — i.e. today's auto behavior.
    # 4. No default_mode (legacy conversations / old clients): unchanged auto rule.
    mode = request.mode
    if mode == "auto":
        default_mode = conversation.get("metadata", {}).get("default_mode")
        if default_mode == "chat":
            mode = "chat"
        else:
            # default_mode == "council" and legacy (no default_mode) both use
            # this rule: council on the effectively-first turn, chat after.
            mode = "council" if effective_message_count == 0 else "chat"
    validate_advanced_settings_for_mode(mode, request)
    ensure_budget_allows_new_turn(conversation_id, conversation)

    return conversation, mode, zdr_enabled, thinking_effort, is_first_message


@app.post("/api/conversations/{conversation_id}/message")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and run the 3-stage council process OR chat with chairman.

    Collects the unified pipeline's events (turn_pipeline.run_turn) into the
    legacy single-JSON response; the stream endpoint serves the same events
    incrementally.
    """
    conversation, mode, zdr_enabled, thinking_effort, is_first_message = prepare_turn(
        conversation_id, request
    )

    collected: Dict[str, Any] = {}
    async for event in run_turn(
        conversation_id,
        request,
        conversation=conversation,
        mode=mode,
        zdr_enabled=zdr_enabled,
        thinking_effort=thinking_effort,
        is_first_message=is_first_message,
    ):
        event_type = event.get("type")
        data = event.get("data")
        if event_type == "error":
            raise HTTPException(status_code=500, detail=event.get("message") or "Turn failed")
        if event_type == "steward_complete":
            collected["evidence"] = data
            collected["steward_usage"] = event.get("usage")
        elif event_type == "stage1_complete":
            collected["stage1"] = data
        elif event_type == "stage2_complete":
            collected["stage2"] = data
            collected["stage2_metadata"] = event.get("metadata") or {}
        elif event_type == "stage3_complete":
            collected["stage3"] = data
        elif event_type == "run_plan":
            collected["run_plan"] = data
        elif event_type == "chat_response":
            collected["chat_response"] = data
        elif event_type == "complete":
            collected["complete"] = data

    complete = collected.get("complete", {})

    if mode == "council":
        stage2_metadata = collected.get("stage2_metadata", {})
        chairman_model = resolve_chairman_model_for_request(
            conversation.get("metadata", {}).get("chairman_model"),
            request,
        )
        metadata = {
            "label_to_model": stage2_metadata.get("label_to_model", {}),
            "aggregate_rankings": stage2_metadata.get("aggregate_rankings", []),
            "steward_usage": collected.get("steward_usage"),
            "steward_model": chairman_model or config.CHAIRMAN_MODEL,
        }
        return {
            "type": "council",
            "stage1": collected.get("stage1", []),
            "stage2": collected.get("stage2", []),
            "stage3": collected.get("stage3", {}),
            "metadata": metadata,
            "evidence": collected.get("evidence"),
            "turn_cost": complete.get("turn_cost", 0.0),
            "total_cost": complete.get("total_cost", 0.0),
            "session_usage": complete.get("session_usage", {}),
            "budget_spent_pct": complete.get("budget_spent_pct"),
        }

    response_dict = collected.get("chat_response", {})
    return {
        "type": "chat",
        "content": response_dict.get("content", ""),
        "reasoning": response_dict.get("reasoning"),
        "turn_cost": complete.get("turn_cost", 0.0),
        "total_cost": complete.get("total_cost", 0.0),
        "session_usage": complete.get("session_usage", {}),
        "budget_spent_pct": complete.get("budget_spent_pct"),
        "run_plan": collected.get("run_plan"),
    }


@app.post("/api/conversations/{conversation_id}/message/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message and stream the response (Council or Chat).
    """
    conversation, mode, zdr_enabled, thinking_effort, is_first_message = prepare_turn(
        conversation_id, request
    )

    async def event_generator():
        async for event in run_turn(
            conversation_id,
            request,
            conversation=conversation,
            mode=mode,
            zdr_enabled=zdr_enabled,
            thinking_effort=thinking_effort,
            is_first_message=is_first_message,
        ):
            yield encode_sse_event(event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )



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
    if use_zdr and not config.provider_is_openrouter():
        raise HTTPException(
            status_code=400,
            detail="ZDR requires OpenRouter. Disable ZDR for this upload or switch providers.",
        )

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


@app.delete("/api/attachments/{attachment_id}")
async def delete_attachment_endpoint(attachment_id: str, force: bool = False):
    """
    Delete an unreferenced attachment's raw file, extracted text, metadata, and cache entry.
    """
    result = delete_attachment(attachment_id, force=force)
    if not result["found"]:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return result


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
    if use_zdr and not config.provider_is_openrouter():
        raise HTTPException(
            status_code=400,
            detail="ZDR requires OpenRouter. Disable ZDR for this upload or switch providers.",
        )

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
