"""JSON-based storage for conversations."""

import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR, SESSION_POLICY_DEFAULTS


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

    # Save to file
    path = get_conversation_path(conversation_id)
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)

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
    with open(path, 'w') as f:
        json.dump(conversation, f, indent=2)


def list_conversations() -> List[Dict[str, Any]]:
    """
    List all conversations (metadata only).

    Returns:
        List of conversation metadata dicts
    """
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                data = json.load(f)
                # Return metadata only
                conversations.append({
                    "id": data["id"],
                    "created_at": data["created_at"],
                    "title": data.get("title", "New Conversation"),
                    "message_count": len(data["messages"])
                })

    # Sort by creation time, newest first
    conversations.sort(key=lambda x: x["created_at"], reverse=True)

    return conversations


def add_user_message(conversation_id: str, content: str):
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

        conversation["messages"].append({
            "role": "user",
            "content": content
        })

        save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    metadata: Dict[str, Any] = None
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

        conversation["messages"].append({
            "role": "assistant",
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "metadata": metadata or {}
        })

        save_conversation(conversation)


def add_chat_message(conversation_id: str, content: str):
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

        conversation["messages"].append({
            "role": "assistant",
            "content": content
        })

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
        
        conversation["session_policy"] = policy
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


def check_budget_warning(conversation_id: str) -> Optional[float]:
    """
    Check if a budget warning should be emitted.
    
    Returns:
        Warning threshold (0.70, 0.85, 1.00) to emit, or None if no warning needed.
        Only returns a threshold that hasn't been warned about yet.
    """
    policy = get_session_policy(conversation_id)
    usage = get_session_usage(conversation_id)
    
    budget = policy.get("budget_usd")
    if budget is None or budget <= 0:
        return None  # No budget set
    
    spent = usage.get("spent_usd", 0.0)
    spent_pct = spent / budget
    last_warning = usage.get("last_warning_level")
    
    thresholds = policy.get("notify_thresholds", [0.70, 0.85, 1.00])
    
    for threshold in thresholds:
        if spent_pct >= threshold:
            # Check if already warned at this level
            if last_warning is None or threshold > last_warning:
                return threshold
    
    return None


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
