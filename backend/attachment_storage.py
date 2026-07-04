"""
Attachment storage and lifecycle management.

Handles file uploads, extraction job tracking, and artifact caching.
Uses SHA256 for deduplication - same file content reuses existing extraction.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from .config import DATA_DIR
from .logger import logger


# Storage directories
ATTACHMENTS_DIR = os.path.join(DATA_DIR, "attachments")
ATTACHMENTS_META_DIR = os.path.join(ATTACHMENTS_DIR, "meta")
ATTACHMENTS_RAW_DIR = os.path.join(ATTACHMENTS_DIR, "raw")
ATTACHMENTS_TEXT_DIR = os.path.join(ATTACHMENTS_DIR, "text")

# SHA256 -> attachment_id cache file
CACHE_INDEX_PATH = os.path.join(ATTACHMENTS_DIR, "cache_index.json")


class AttachmentStats(BaseModel):
    """Statistics about extracted content."""
    page_count: Optional[int] = None
    sheet_count: Optional[int] = None
    slide_count: Optional[int] = None
    char_count: int = 0
    empty_page_ratio: float = 0.0


class Attachment(BaseModel):
    """Attachment metadata model."""
    attachment_id: str
    sha256: str
    filename: str
    mime_type: str
    size_bytes: int
    created_at: str
    status: str = "processing"  # processing, success, partial, failed
    method: str = "local"  # local, openrouter_pdf_text, openrouter_mistral_ocr, vision
    warning: Optional[str] = None
    error: Optional[str] = None
    stats: AttachmentStats = Field(default_factory=AttachmentStats)
    conversation_ids: List[str] = Field(default_factory=list)


def ensure_dirs():
    """Ensure attachment directories exist."""
    for d in [ATTACHMENTS_DIR, ATTACHMENTS_META_DIR, ATTACHMENTS_RAW_DIR, ATTACHMENTS_TEXT_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def compute_sha256(content: bytes) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def get_cache_index() -> Dict[str, str]:
    """Load SHA256 -> attachment_id cache index."""
    ensure_dirs()
    if os.path.exists(CACHE_INDEX_PATH):
        with open(CACHE_INDEX_PATH, 'r', encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache_index(index: Dict[str, str]):
    """Save SHA256 -> attachment_id cache index."""
    ensure_dirs()
    with open(CACHE_INDEX_PATH, 'w', encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def get_cached_attachment(sha256: str) -> Optional[str]:
    """Check if file with this hash was already processed.
    
    Returns:
        attachment_id if cached, None otherwise
    """
    index = get_cache_index()
    attachment_id = index.get(sha256)
    
    if attachment_id:
        # Verify the attachment still exists
        meta = get_attachment(attachment_id)
        if meta and meta.status in ("success", "partial"):
            logger.info(f"[ATTACH] Cache hit for {sha256[:16]}... -> {attachment_id}")
            return attachment_id
    
    return None


def create_attachment(
    content: bytes,
    filename: str,
    mime_type: str
) -> Attachment:
    """Create a new attachment from file content.
    
    This stores the raw file and initializes metadata.
    Extraction happens asynchronously via process_attachment().
    
    Returns:
        Attachment metadata (status will be 'processing')
    """
    ensure_dirs()
    
    sha256 = compute_sha256(content)
    
    # Check cache first
    cached_id = get_cached_attachment(sha256)
    if cached_id:
        cached = get_attachment(cached_id)
        if cached:
            logger.info(f"[ATTACH] Reusing cached attachment {cached_id} for {filename}")
            return cached
    
    # Create new attachment
    attachment_id = f"att_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    
    attachment = Attachment(
        attachment_id=attachment_id,
        sha256=sha256,
        filename=filename,
        mime_type=mime_type,
        size_bytes=len(content),
        created_at=now,
        status="processing",
        method="local",
    )
    
    # Save raw file
    raw_path = os.path.join(ATTACHMENTS_RAW_DIR, f"{attachment_id}.bin")
    with open(raw_path, 'wb') as f:
        f.write(content)
    
    # Save metadata
    save_attachment(attachment)
    
    # Update cache index
    index = get_cache_index()
    index[sha256] = attachment_id
    save_cache_index(index)
    
    logger.info(f"[ATTACH] Created attachment {attachment_id} for {filename} ({len(content)} bytes)")
    
    return attachment


def get_attachment(attachment_id: str) -> Optional[Attachment]:
    """Load attachment metadata by ID."""
    meta_path = os.path.join(ATTACHMENTS_META_DIR, f"{attachment_id}.json")
    
    if not os.path.exists(meta_path):
        return None
    
    with open(meta_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    
    return Attachment(**data)


def save_attachment(attachment: Attachment):
    """Save attachment metadata."""
    ensure_dirs()
    meta_path = os.path.join(ATTACHMENTS_META_DIR, f"{attachment.attachment_id}.json")
    
    with open(meta_path, 'w', encoding="utf-8") as f:
        json.dump(attachment.model_dump(), f, indent=2)


def link_attachments_to_conversation(
    attachment_ids: List[str],
    conversation_id: str,
) -> List[Attachment]:
    """Mark attachments as referenced by a conversation and return metadata."""
    linked = []
    updated = set()

    for attachment_id in attachment_ids:
        attachment = get_attachment(attachment_id)
        if not attachment:
            continue

        if attachment_id not in updated:
            if conversation_id not in attachment.conversation_ids:
                attachment.conversation_ids.append(conversation_id)
                save_attachment(attachment)
            updated.add(attachment_id)

        linked.append(attachment)

    return linked


def collect_attachment_ids_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    """Collect unique attachment IDs from persisted conversation messages."""
    attachment_ids = []
    seen = set()

    def add_id(attachment_id: Optional[str]):
        if attachment_id and attachment_id not in seen:
            seen.add(attachment_id)
            attachment_ids.append(attachment_id)

    for message in messages:
        for attachment_id in message.get("attachment_ids") or []:
            add_id(attachment_id)

        for attachment in message.get("attachments") or []:
            if isinstance(attachment, dict):
                add_id(attachment.get("attachment_id"))

    return attachment_ids


def _remove_from_cache_index(attachment: Attachment):
    index = get_cache_index()
    changed = False

    for sha256, attachment_id in list(index.items()):
        if sha256 == attachment.sha256 or attachment_id == attachment.attachment_id:
            del index[sha256]
            changed = True

    if changed:
        save_cache_index(index)


def _delete_file(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def delete_attachment(
    attachment_id: str,
    conversation_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Delete an attachment's raw/text/meta artifacts when it has no remaining refs."""
    attachment = get_attachment(attachment_id)
    if not attachment:
        return {
            "attachment_id": attachment_id,
            "found": False,
            "deleted": False,
            "retained": False,
            "conversation_ids": [],
            "files_deleted": 0,
        }

    remaining_refs = list(attachment.conversation_ids)
    if conversation_id:
        remaining_refs = [cid for cid in remaining_refs if cid != conversation_id]

    if remaining_refs and not force:
        attachment.conversation_ids = remaining_refs
        save_attachment(attachment)
        logger.info(
            "[ATTACH] Retained %s; still referenced by %d conversation(s)",
            attachment_id,
            len(remaining_refs),
        )
        return {
            "attachment_id": attachment_id,
            "found": True,
            "deleted": False,
            "retained": True,
            "conversation_ids": remaining_refs,
            "files_deleted": 0,
        }

    raw_path = os.path.join(ATTACHMENTS_RAW_DIR, f"{attachment_id}.bin")
    text_path = os.path.join(ATTACHMENTS_TEXT_DIR, f"{attachment_id}.txt")
    meta_path = os.path.join(ATTACHMENTS_META_DIR, f"{attachment_id}.json")
    files_deleted = sum(_delete_file(path) for path in [raw_path, text_path, meta_path])
    _remove_from_cache_index(attachment)

    logger.info("[ATTACH] Deleted attachment %s (%d artifact files)", attachment_id, files_deleted)
    return {
        "attachment_id": attachment_id,
        "found": True,
        "deleted": True,
        "retained": False,
        "conversation_ids": [],
        "files_deleted": files_deleted,
    }


def delete_attachments_for_conversation(
    conversation_id: str,
    conversation: Dict[str, Any],
) -> Dict[str, Any]:
    """Unreference and delete unshared attachments used by a conversation."""
    attachment_ids = collect_attachment_ids_from_messages(conversation.get("messages", []))
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


def update_attachment_status(
    attachment_id: str,
    status: str,
    method: str = None,
    warning: str = None,
    error: str = None,
    stats: Dict[str, Any] = None
):
    """Update attachment status after extraction."""
    attachment = get_attachment(attachment_id)
    if not attachment:
        raise ValueError(f"Attachment {attachment_id} not found")
    
    attachment.status = status
    if method:
        attachment.method = method
    if warning:
        attachment.warning = warning
    if error:
        attachment.error = error
    if stats:
        attachment.stats = AttachmentStats(**stats)
    
    save_attachment(attachment)
    logger.info(f"[ATTACH] Updated {attachment_id} status={status} method={method}")


def save_attachment_text(attachment_id: str, text: str):
    """Save extracted text for an attachment."""
    ensure_dirs()
    text_path = os.path.join(ATTACHMENTS_TEXT_DIR, f"{attachment_id}.txt")
    
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_attachment_text(attachment_id: str) -> Optional[str]:
    """Load extracted text for an attachment."""
    text_path = os.path.join(ATTACHMENTS_TEXT_DIR, f"{attachment_id}.txt")
    
    if not os.path.exists(text_path):
        return None
    
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_attachment_raw(attachment_id: str) -> Optional[bytes]:
    """Load raw file content for an attachment."""
    raw_path = os.path.join(ATTACHMENTS_RAW_DIR, f"{attachment_id}.bin")
    
    if not os.path.exists(raw_path):
        return None
    
    with open(raw_path, 'rb') as f:
        return f.read()


def list_attachments(attachment_ids: List[str]) -> List[Attachment]:
    """Load multiple attachments by ID."""
    result = []
    for aid in attachment_ids:
        att = get_attachment(aid)
        if att:
            result.append(att)
    return result


def build_llm_context(attachment_ids: List[str], max_chars: int = 50000) -> str:
    """Build LLM context from attachments.
    
    This is what gets injected into the LLM prompt, hidden from the user.
    Includes file content with clear attribution for citations.
    
    Args:
        attachment_ids: List of attachment IDs to include
        max_chars: Maximum total characters (to stay within token limits)
    
    Returns:
        Formatted context string for LLM
    """
    if not attachment_ids:
        return ""
    
    attachments = list_attachments(attachment_ids)
    if not attachments:
        return ""
    
    context_parts = []
    remaining_chars = max_chars
    
    for att in attachments:
        if att.status not in ("success", "partial"):
            # Include a note about failed extractions
            context_parts.append(f"[File: {att.filename}]\nStatus: Extraction {att.status}. {att.error or att.warning or 'No content available.'}\n")
            continue
        
        text = get_attachment_text(att.attachment_id)
        if not text:
            continue
        
        # Build attribution header
        header = f"[File: {att.filename}]"
        if att.stats.page_count:
            header += f" ({att.stats.page_count} pages)"
        elif att.stats.slide_count:
            header += f" ({att.stats.slide_count} slides)"
        elif att.stats.sheet_count:
            header += f" ({att.stats.sheet_count} sheets)"
        
        # Truncate if needed
        available = remaining_chars - len(header) - 10  # Reserve for newlines
        if available <= 0:
            break
        
        if len(text) > available:
            text = text[:available] + "\n[...content truncated...]"
        
        context_parts.append(f"{header}\n{text}\n")
        remaining_chars -= len(context_parts[-1])
    
    if not context_parts:
        return ""
    
    return "--- ATTACHED DOCUMENTS ---\n\n" + "\n".join(context_parts) + "\n--- END DOCUMENTS ---"
