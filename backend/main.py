"""FastAPI backend for LLM Council."""

from fastapi import FastAPI, HTTPException
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

# Initialize RAG system
rag_system = CouncilRAG()

def get_turn_index(conversation: Dict[str, Any]) -> int:
    """Count the number of completed Council turns (messages with stage3)."""
    count = 0
    for msg in conversation.get("messages", []):
        if msg.get("role") == "assistant" and "stage3" in msg:
            count += 1
    return count

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
    pass


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


@app.get("/api/conversations", response_model=List[ConversationMetadata])
async def list_conversations():
    """List all conversations (metadata only)."""
    return storage.list_conversations()


@app.post("/api/conversations", response_model=Conversation)
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    conversation_id = str(uuid.uuid4())
    conversation = storage.create_conversation(conversation_id)
    return conversation


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
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

    if mode == "council":
        # Run the 3-stage council process
        stage1_results, stage2_results, stage3_result, metadata = await run_full_council(
            request.content
        )

        # Add assistant message with all stages
        storage.add_assistant_message(
            conversation_id,
            stage1_results,
            stage2_results,
            stage3_result
        )

        # Index the session for RAG
        updated_conversation = storage.get_conversation(conversation_id)
        turn_index = get_turn_index(updated_conversation) - 1
        
        rag_system.index_session(
            conversation_id,
            turn_index,
            request.content,
            stage1_results,
            stage2_results,
            stage3_result
        )

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
        
        # Retrieve context via RAG
        rag_context = rag_system.retrieve(request.content, conversation_id)
        
        response_dict = await chat_with_chairman(
            request.content,
            conversation["messages"],
            rag_context
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

            if mode == "council":
                # Start title generation in parallel (don't await yet)
                title_task = None
                if is_first_message:
                    title_task = asyncio.create_task(generate_conversation_title(request.content))

                # Stage 1: Collect responses
                yield f"data: {json.dumps({'type': 'stage1_start'})}\n\n"
                stage1_results = await stage1_collect_responses(request.content)
                yield f"data: {json.dumps({'type': 'stage1_complete', 'data': stage1_results})}\n\n"

                # Stage 2: Collect rankings
                yield f"data: {json.dumps({'type': 'stage2_start'})}\n\n"
                stage2_results, label_to_model = await stage2_collect_rankings(request.content, stage1_results)
                aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
                yield f"data: {json.dumps({'type': 'stage2_complete', 'data': stage2_results, 'metadata': {'label_to_model': label_to_model, 'aggregate_rankings': aggregate_rankings}})}\n\n"

                # Stage 3: Synthesize final answer
                yield f"data: {json.dumps({'type': 'stage3_start'})}\n\n"
                stage3_result = await stage3_synthesize_final(request.content, stage1_results, stage2_results)
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

                # Index for RAG
                updated_conversation = storage.get_conversation(conversation_id)
                turn_index = get_turn_index(updated_conversation) - 1
                
                rag_system.index_session(
                    conversation_id,
                    turn_index,
                    request.content,
                    stage1_results,
                    stage2_results,
                    stage3_result
                )
            
            else:
                # Chat mode
                yield f"data: {json.dumps({'type': 'chat_start'})}\n\n"
                
                # Reload conversation to get history
                updated_conversation = storage.get_conversation(conversation_id)
                
                # Retrieve context via RAG
                rag_context = rag_system.retrieve(request.content, conversation_id)
                
                response_dict = await chat_with_chairman(
                    request.content,
                    updated_conversation["messages"],
                    rag_context
                )
                
                # Save chat message
                storage.add_chat_message(conversation_id, response_dict["content"])
                
                yield f"data: {json.dumps({'type': 'chat_response', 'data': response_dict})}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
