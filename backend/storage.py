"""JSON-based storage for conversations."""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR, SESSION_POLICY_DEFAULTS
from .logger import logger


class ConversationLock:
    """Thread-safe locking mechanism for conversation access."""
    _locks = {}
    _main_lock = threading.Lock()
    
    @classmethod
    def get_lock(cls, conversation_id: str):
        """Get or create a lock for a specific conversation."""
        with cls._main_lock:
            if conversation_id not in cls._locks:
                cls._locks[conversation_id] = threading.Lock()
            return cls._locks[conversation_id]

FOLDER_LOCK = threading.Lock()


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation."""
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def create_conversation(conversation_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        conversation_id: Unique identifier for the conversation
        metadata: Optional metadata to store (e.g., selected models)

    Returns:
        New conversation dict
    """
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": [],
        "metadata": metadata or {},
        "total_cost": 0.0
    }

    # Save to file atomically
    path = get_conversation_path(conversation_id)
    temp_path = path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(conversation, f, indent=2)
    os.replace(temp_path, path)

    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a conversation from storage.

    Args:
        conversation_id: Unique identifier for the conversation

    Returns:
        Conversation dict or None if not found
    """
    path = get_conversation_path(conversation_id)

    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        return json.load(f)


def save_conversation(conversation: Dict[str, Any]):
    """
    Save a conversation to storage.

    Args:
        conversation: Conversation dict to save
    """
    ensure_data_dir()

    path = get_conversation_path(conversation['id'])
    temp_path = path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(conversation, f, indent=2)
    os.replace(temp_path, path)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json') and filename != 'folders.json':
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Return metadata only
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "title": data.get("title", "New Conversation"),
                    "message_count": len(data["messages"]),
                    "folder_id": data.get("metadata", {}).get("folder_id")
                })

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(
    conversation_id: str,
    content: str,
    attachment_ids: Optional[List[str]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
):
    """
    Add a user message to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: User message content
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = {
            "role": "user",
            "content": content
        }
        if attachment_ids:
            message["attachment_ids"] = attachment_ids
        if attachments:
            message["attachments"] = attachments

        conversation["messages"].append(message)

        save_conversation(conversation)


def truncate_messages(conversation_id: str, keep_count: int) -> Dict[str, Any]:
    """
    Truncate a conversation's messages, keeping only the first `keep_count` messages.
    Used by Edit & Regenerate to discard messages after the edit point.

    Args:
        conversation_id: Conversation identifier
        keep_count: Number of messages to keep (0-indexed, exclusive)

    Returns:
        The updated conversation dict
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        original_count = len(conversation["messages"])
        conversation["messages"] = conversation["messages"][:keep_count]
        save_conversation(conversation)

        logger.info(
            "[EDIT] Truncated conversation %s from %d to %d messages",
            conversation_id, original_count, keep_count
        )
        return conversation


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    metadata: Dict[str, Any] = None,
    running_cost: float = None,
):
    """
    Add an assistant message with all 3 stages to a conversation.

    Args:
        conversation_id: Conversation identifier
        stage1: List of individual model responses
        stage2: List of model rankings
        stage3: Final synthesized response
        metadata: Optional metadata including label_to_model mapping for analytics
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = {
            "role": "assistant",
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "metadata": metadata or {},
        }
        if running_cost is not None:
            message["running_cost"] = running_cost

        conversation["messages"].append(message)

        save_conversation(conversation)


def add_chat_message(conversation_id: str, content: str, running_cost: float = None):
    """
    Add a simple chat message (from assistant) to a conversation.

    Args:
        conversation_id: Conversation identifier
        content: The assistant's response text
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = {
            "role": "assistant",
            "content": content,
        }
        if running_cost is not None:
            message["running_cost"] = running_cost

        conversation["messages"].append(message)

        save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """
    Update the title of a conversation.

    Args:
        conversation_id: Conversation identifier
        title: New title for the conversation
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation["title"] = title
        save_conversation(conversation)


def update_conversation_cost(conversation_id: str, cost: float):
    """
    Update the total cost of a conversation.

    Args:
        conversation_id: Conversation identifier
        cost: Cost to add to the total
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        current_cost = conversation.get("total_cost", 0.0)
        conversation["total_cost"] = current_cost + cost
        save_conversation(conversation)


# =============================================================================
# SESSION BUDGET FUNCTIONS
# =============================================================================

def get_session_policy(conversation_id: str) -> Dict[str, Any]:
    """
    Get the session policy for a conversation.
    Returns defaults if not set.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return SESSION_POLICY_DEFAULTS.copy()
    
    policy = conversation.get("session_policy", {})
    # Merge with defaults for any missing keys
    return {**SESSION_POLICY_DEFAULTS, **policy}


def set_session_policy(conversation_id: str, policy: Dict[str, Any]):
    """
    Set the session policy for a conversation.
    
    Args:
        policy: Dict with budget_usd, notify_thresholds, mode, allow_overage
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation["session_policy"] = {**SESSION_POLICY_DEFAULTS, **policy}
        save_conversation(conversation)


def get_session_usage(conversation_id: str) -> Dict[str, Any]:
    """
    Get the current session usage for a conversation.
    Returns initialized usage if not set.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return {"spent_usd": 0.0, "messages": 0, "last_warning_level": None}
    
    return conversation.get("session_usage", {
        "spent_usd": 0.0,
        "messages": 0,
        "last_warning_level": None
    })


def update_session_usage(conversation_id: str, cost_delta: float, emit_warning: float = None):
    """
    Update session usage after a message.
    
    Args:
        conversation_id: Conversation identifier
        cost_delta: Cost to add to spent_usd
        emit_warning: Warning threshold level to record (0.70, 0.85, 1.00), or None
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        usage = conversation.get("session_usage", {
            "spent_usd": 0.0,
            "messages": 0,
            "last_warning_level": None
        })
        
        usage["spent_usd"] = usage.get("spent_usd", 0.0) + cost_delta
        usage["messages"] = usage.get("messages", 0) + 1
        
        if emit_warning is not None:
            usage["last_warning_level"] = emit_warning
        
        conversation["session_usage"] = usage
        save_conversation(conversation)


def _get_new_warning_level(policy: Dict[str, Any], usage: Dict[str, Any]) -> Optional[float]:
    budget = policy.get("budget_usd")
    if budget is None or budget <= 0:
        return None

    spent_pct = usage.get("spent_usd", 0.0) / budget
    last_warning = usage.get("last_warning_level")
    thresholds = sorted(policy.get("notify_thresholds", [0.70, 0.85, 1.00]))

    crossed = [
        threshold
        for threshold in thresholds
        if spent_pct >= threshold and (last_warning is None or threshold > last_warning)
    ]
    if not crossed:
        return None
    return max(crossed)


def record_session_usage(conversation_id: str, cost_delta: float) -> Dict[str, Any]:
    """
    Add a turn's cost to session usage and return the updated budget state.
    Warning calculation happens after the current turn cost is included.
    """
    with ConversationLock.get_lock(conversation_id):
        conversation = get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        policy = {**SESSION_POLICY_DEFAULTS, **conversation.get("session_policy", {})}
        usage = conversation.get("session_usage", {
            "spent_usd": 0.0,
            "messages": 0,
            "last_warning_level": None,
        })

        usage["spent_usd"] = usage.get("spent_usd", 0.0) + cost_delta
        usage["messages"] = usage.get("messages", 0) + 1

        warning_level = _get_new_warning_level(policy, usage)
        if warning_level is not None:
            usage["last_warning_level"] = warning_level

        conversation["session_usage"] = usage
        save_conversation(conversation)

    budget = policy.get("budget_usd")
    budget_spent_pct = None
    if budget is not None and budget > 0:
        budget_spent_pct = usage.get("spent_usd", 0.0) / budget

    return {
        "usage": usage.copy(),
        "warning_level": warning_level,
        "budget_spent_pct": budget_spent_pct,
    }


def check_budget_warning(conversation_id: str) -> Optional[float]:
    """
    Check if a budget warning should be emitted.
    
    Returns:
        Warning threshold (0.70, 0.85, 1.00) to emit, or None if no warning needed.
        Only returns a threshold that hasn't been warned about yet.
    """
    policy = get_session_policy(conversation_id)
    usage = get_session_usage(conversation_id)
    
    return _get_new_warning_level(policy, usage)


def get_budget_spent_percentage(conversation_id: str) -> Optional[float]:
    """
    Get the percentage of budget spent.
    
    Returns:
        Float 0-N (can exceed 1.0 if over budget), or None if no budget set.
    """
    policy = get_session_policy(conversation_id)
    usage = get_session_usage(conversation_id)
    
    budget = policy.get("budget_usd")
    if budget is None or budget <= 0:
        return None
    
    spent = usage.get("spent_usd", 0.0)
    return spent / budget


# =============================================================================
# CHAT MANAGEMENT (PHASE 5)
# =============================================================================

def delete_conversation(conversation_id: str) -> bool:
    """
    Atomically delete a conversation file.
    Note: Vector DB cleanup should be triggered by the caller.
    """
    path = get_conversation_path(conversation_id)
    if not os.path.exists(path):
        return False
        
    lock = ConversationLock.get_lock(conversation_id)
    with lock:
        try:
            os.remove(path)
            return True
        except OSError:
            return False

def update_conversation_folder(conversation_id: str, folder_id: Optional[str]) -> bool:
    """Move a conversation to a specific folder (or remove from folder if None)."""
    lock = ConversationLock.get_lock(conversation_id)
    with lock:
        conv = get_conversation(conversation_id)
        if not conv:
            return False
            
        if "metadata" not in conv:
            conv["metadata"] = {}
            
        if folder_id is None:
            conv["metadata"].pop("folder_id", None)
        else:
            conv["metadata"]["folder_id"] = folder_id
            
        save_conversation(conv)
        return True


# =============================================================================
# FOLDER MANAGEMENT (PHASE 5)
# =============================================================================

def get_folders_path() -> str:
    return os.path.join(DATA_DIR, "folders.json")


def _read_folders() -> List[Dict[str, Any]]:
    path = get_folders_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def _write_folders(folders: List[Dict[str, Any]]):
    ensure_data_dir()
    path = get_folders_path()
    temp_path = path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(folders, f, indent=2)
    os.replace(temp_path, path)


def list_folders() -> List[Dict[str, Any]]:
    """Get all folders sorted by creation time."""
    with FOLDER_LOCK:
        folders = _read_folders()
        folders.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return folders


def create_folder(folder_id: str, name: str, color: str = None) -> Dict[str, Any]:
    """Create a new folder."""
    with FOLDER_LOCK:
        folders = _read_folders()
        new_folder = {
            "id": folder_id,
            "name": name,
            "color": color,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        folders.append(new_folder)
        _write_folders(folders)
        return new_folder


def update_folder(folder_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a folder's properties."""
    with FOLDER_LOCK:
        folders = _read_folders()
        for f in folders:
            if f["id"] == folder_id:
                if "name" in updates and updates["name"]:
                    f["name"] = updates["name"]
                if "color" in updates:
                    f["color"] = updates["color"]
                f["updated_at"] = datetime.utcnow().isoformat()
                _write_folders(folders)
                return f
        return None


def delete_folder(folder_id: str) -> bool:
    """Delete a folder. Note: Conversations inside will be orphaned (folder_id points to nothing), which is fine for the frontend to handle as 'root'."""
    with FOLDER_LOCK:
        folders = _read_folders()
        new_folders = [f for f in folders if f["id"] != folder_id]
        if len(new_folders) == len(folders):
            return False
        _write_folders(new_folders)
        return True
