"""Bounded, versioned reads of registered materials. No model-writable paths."""
import hashlib
import json
import re
from itertools import chain

from . import attachment_storage, storage

SEGMENT_VERSION = "paragraph-page-v1"
MAX_CHARS = 50_000
CHUNK_CHARS = 2_000
SOURCE_ID = re.compile(r"[A-Za-z0-9_-]+")
CITATION = re.compile(r"\[S([1-9][0-9]*)\.([1-9][0-9]*)\]")
CANDIDATE = re.compile(r"\[S[^\]\n]*(?:\]|(?=\n|$))")


def source_version(text):
    return hashlib.sha256((SEGMENT_VERSION + "\0" + text).encode("utf-8")).hexdigest()


def material_chunks(text, *, pdf=False):
    """Offsets refer to the unchanged extracted text, including page markers."""
    page = None
    ordinal = 0
    boundary = re.compile(r"(?m)^\[Page ([1-9][0-9]*)\]\s*\n|\n[ \t]*\n" if pdf else r"\n[ \t]*\n")
    start = 0
    for match in chain(boundary.finditer(text), (None,)):
        end = match.start() if match else len(text)
        for offset in range(start, end, CHUNK_CHARS):
            chunk_end = min(offset + CHUNK_CHARS, end)
            if text[offset:chunk_end].strip():
                ordinal += 1
                yield {"ordinal": ordinal, "page": page, "start": offset, "end": chunk_end, "text": text[offset:chunk_end]}
        if match:
            if pdf and match.group(1):
                page = int(match.group(1))
            start = match.end()


def validate_new_attachment_ids(conversation_id, attachment_ids):
    """An opaque ID alone cannot adopt another conversation's registered source."""
    for source_id in attachment_ids:
        if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
            raise ValueError("Invalid material ID")
        attachment = attachment_storage.get_attachment(source_id)
        if attachment and attachment.conversation_ids and conversation_id not in attachment.conversation_ids:
            raise PermissionError("Material belongs to another conversation; upload the file for this conversation")


def resolve_source_scope(conversation, attachment_ids=(), selected=None, edit_index=-1):
    """Rebuild membership from the retained messages, never stale metadata links."""
    validate_new_attachment_ids(conversation.get("id"), attachment_ids)
    messages = conversation.get("messages", [])
    if edit_index >= 0:
        messages = messages[:edit_index]
    candidates = list(dict.fromkeys([
        *attachment_storage.collect_attachment_ids_from_messages(messages), *attachment_ids,
    ]))
    if any(not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id) for source_id in candidates):
        raise ValueError("Invalid material ID")
    if selected is not None:
        if any(source_id not in candidates for source_id in selected):
            raise PermissionError("Selected materials must belong to this conversation or this upload")
        candidates = [source_id for source_id in candidates if source_id in selected]
    sources = []
    for source_id in candidates:
        attachment = attachment_storage.get_attachment(source_id)
        text = attachment_storage.get_attachment_text(source_id) if attachment and attachment.status in ("success", "partial") else None
        sources.append({
            "source_id": source_id,
            "title": attachment.filename if attachment else source_id,
            "status": attachment.status if attachment else "missing",
            "version_id": source_version(text) if text and text.strip() else None,
            "warning": (attachment.warning or attachment.error) if attachment else "Material no longer exists",
        })
    return sources


def read_material(source_id, version_id, chunk_id=None, *, allowed_source_ids, conversation_id=None):
    """The caller owns allowed_source_ids. MCP and AAB share this read boundary."""
    if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id) or source_id not in allowed_source_ids:
        raise PermissionError("Material is outside the allowed source scope")
    if conversation_id is not None:
        conversation = storage.get_conversation(conversation_id)
        if not conversation or source_id not in attachment_storage.collect_attachment_ids_from_messages(conversation.get("messages", [])):
            raise PermissionError("Material is no longer registered in this conversation")
    attachment = attachment_storage.get_attachment(source_id)
    if not attachment or attachment.status not in ("success", "partial"):
        raise FileNotFoundError("Material is missing or has no readable extraction")
    text = attachment_storage.get_attachment_text(source_id)
    if not text or not text.strip():
        raise FileNotFoundError("Material has no readable extracted text")
    if source_version(text) != version_id:
        raise ValueError("Material version changed; select the current version before reading")
    if chunk_id is not None and (isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 1):
        raise ValueError("chunk_id must be a positive integer")
    chunks = []
    omitted = []
    remaining = MAX_CHARS
    count = 0
    for chunk in material_chunks(text, pdf=attachment.mime_type == "application/pdf"):
        count = chunk["ordinal"]
        if (chunk_id is None or chunk_id == count) and len(chunk["text"]) <= remaining:
            chunks.append(chunk)
            remaining -= len(chunk["text"])
        else:
            omitted.append(count)
    if chunk_id is not None and chunk_id > count:
        raise ValueError("Unknown material chunk")
    return {"source_id": source_id, "version_id": version_id, "status": attachment.status,
            "chunks": chunks, "total_chunks": count, "omitted": omitted}


def build_evidence_snapshot(sources, *, conversation_id=None):
    allowed = {source["source_id"] for source in sources if source["version_id"]}
    snapshot = {"schema_version": 1, "segment_version": SEGMENT_VERSION, "sources": [], "citations": {}}
    remaining = MAX_CHARS
    for index, source in enumerate(sources, 1):
        item = {**source, "alias": f"S{index}", "included": [], "omitted": []}
        snapshot["sources"].append(item)
        if not source["version_id"]:
            continue
        result = read_material(source["source_id"], source["version_id"], allowed_source_ids=allowed, conversation_id=conversation_id)
        item["total_chunks"] = result["total_chunks"]
        item["omitted"] = list(result["omitted"])
        for chunk in result["chunks"]:
            if len(chunk["text"]) > remaining:
                item["omitted"].append(chunk["ordinal"])
                continue
            remaining -= len(chunk["text"])
            item["included"].append(chunk["ordinal"])
            token = f"[S{index}.{chunk['ordinal']}]"
            snapshot["citations"][token] = {
                "source_id": source["source_id"], "version_id": source["version_id"], **chunk,
            }
        item["omitted"].sort()
    snapshot["content_hash"] = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return snapshot


def validate_citations(text, snapshot):
    checks = []
    for match in CANDIDATE.finditer(text):
        token = match.group()
        status = "malformed" if not CITATION.fullmatch(token) else (
            "valid_locator" if token in snapshot.get("citations", {}) else "invalid_id"
        )
        checks.append({"token": token, "start": match.start(), "end": match.end(), "status": status})
    return {"items": checks, "no_citations": not checks}


def revalidate_evidence(snapshot, conversation_id):
    allowed = {source["source_id"] for source in snapshot["sources"] if source.get("included")}
    for source in snapshot["sources"]:
        if source["source_id"] in allowed:
            current = read_material(source["source_id"], source["version_id"], source["included"][0],
                                    allowed_source_ids=allowed, conversation_id=conversation_id)
            if current["status"] != source["status"]:
                raise ValueError("Material extraction status changed; select the current extraction")


def historical_citations(text):
    """Change only model-facing projection. The saved answer remains untouched."""
    return CITATION.sub(lambda match: f"(historical citation S{match[1]}.{match[2]}; not current evidence)", text)
