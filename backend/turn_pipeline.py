"""Single turn pipeline shared by the stream and non-stream endpoints (audit §5.1).

Both POST /message and POST /message/stream consume ``run_turn``: the stream
endpoint SSE-encodes each yielded event; the non-stream endpoint collects the
events and assembles its legacy JSON response. There is exactly one
implementation of the council/chat turn logic.

Collaborators (stage functions, storage, RAG, cost helpers) are resolved
through ``backend.main`` at call time so tests that monkeypatch them on the
main module keep working.
"""
import asyncio
import uuid
from typing import Any, AsyncIterator, Dict

from .logger import logger

# Kept in sync with council.run_full_council's all-fail early return.
ALL_FAIL_STAGE3 = {
    "model": "error",
    "response": "All models failed to respond. Please try again.",
}


async def run_turn(
    conversation_id: str,
    request: Any,
    *,
    conversation: Dict[str, Any],
    mode: str,
    zdr_enabled: bool,
    thinking_effort: Any,
    is_first_message: bool,
) -> AsyncIterator[Dict[str, Any]]:
    """Run one conversation turn, yielding normalized event dicts.

    Events end with {'type': 'complete', ...} on success or
    {'type': 'error', 'message': ...} on failure. Validation/mode resolution
    happens in the endpoints (main.prepare_turn) BEFORE this generator runs so
    HTTP errors surface as proper status codes, not mid-stream events.
    """
    from . import main  # late import: honors monkeypatched seams + avoids import cycle
    from . import config

    try:
        current_conversation = conversation

        # Build attachment context if attachment_ids provided
        attachment_context = ""
        has_attachments = bool(request.attachment_ids)
        if has_attachments:
            attachment_context = main.build_llm_context(request.attachment_ids)
            logger.info(f"[ATTACH] Built context from {len(request.attachment_ids)} attachments ({len(attachment_context)} chars)")
            # Index documents into PageIndex for cross-conversation retrieval.
            # Skip entirely for ZDR turns (audit §12, Decision #5): PageIndex
            # memory is cross-conversation, so ZDR content must never enter it.
            if not zdr_enabled:
                for att_id in request.attachment_ids:
                    att_text = main.get_attachment_text(att_id)
                    if att_text:
                        main.rag_system.index_document(conversation_id, att_id, att_text)

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
            attachment_cleanup = main.delete_truncated_message_attachments(
                conversation_id,
                request.edit_index,
            )
            main.storage.truncate_messages(conversation_id, request.edit_index)
            # Re-fetch conversation after truncation
            current_conversation = main.storage.get_conversation(conversation_id)
            yield {"type": "edit_truncated", "data": {"edit_index": request.edit_index, "attachments": attachment_cleanup}}

        message_attachments = main.prepare_message_attachments(conversation_id, request.attachment_ids)

        # Add user message (store only original content, not attachment text)
        main.storage.add_user_message(
            conversation_id,
            request.content,
            attachment_ids=request.attachment_ids,
            attachments=message_attachments,
        )

        # Get model configuration from conversation metadata
        metadata = current_conversation.get("metadata", {})
        council_models = metadata.get("council_models")
        chairman_model = main.resolve_chairman_model_for_request(
            metadata.get("chairman_model"),
            request,
        )

        if mode == "council":
            # Start title generation in parallel (don't await yet)
            title_task = None
            if is_first_message:
                title_task = asyncio.create_task(
                    main.generate_conversation_title(
                        request.content,
                        zdr_enabled=zdr_enabled,
                    )
                )

            # Stage 0a: Web Search (if enabled)
            web_context = ""
            if request.web_search_enabled:
                from .web_search import web_search_stage0
                yield {"type": "web_search_start"}
                search_result = await web_search_stage0(
                    request.content,
                    depth=request.web_search_depth,
                    zdr_enabled=zdr_enabled,
                )
                web_context = search_result.get("context", "")
                if search_result.get("usage"):
                    extra_usage_records.append({
                        "model": search_result.get("model"),
                        "usage": search_result.get("usage", {}),
                    })
                yield {"type": "web_search_complete", "data": {"context": web_context[:500], "citations": search_result.get("citations", []), "model": search_result.get("model", "")}}
                if web_context:
                    llm_content = f"[Web Search Results]\n{web_context}\n\n[User Query]\n{llm_content}"

            # Stage 0b: Tool Steward
            run_id = str(uuid.uuid4())

            yield {"type": "steward_start"}
            evidence_pack, steward_usage = await main.run_tool_steward_phase(
                request.content,
                run_id,
                chairman_model=chairman_model,
                zdr_enabled=zdr_enabled,
            )
            if steward_usage:
                extra_usage_records.append({
                    "model": chairman_model or config.CHAIRMAN_MODEL,
                    "usage": steward_usage,
                })
            yield {"type": "steward_complete", "data": evidence_pack.model_dump(), "usage": steward_usage}

            # Stage 1: Collect responses (use llm_content with attachments)
            yield {"type": "stage1_start"}
            stage1_results = await main.stage1_collect_responses(
                llm_content,
                models=council_models,
                evidence_pack=evidence_pack,
                zdr_enabled=zdr_enabled,
                thinking_effort=thinking_effort,
            )
            for index, result in enumerate(stage1_results):
                for event in main.build_reasoning_stream_events(
                    result,
                    scope="council",
                    stage="stage1",
                    model=result.get("model"),
                    content_key="response",
                    index=index,
                ):
                    yield event
            yield {"type": "stage1_complete", "data": stage1_results}

            if not stage1_results:
                # All models failed: emit a clean error result instead of
                # synthesizing garbage from nothing (parity with
                # council.run_full_council's early return, audit §4.2).
                stage2_results = []
                label_to_model = {}
                aggregate_rankings = []
                stage3_result = dict(ALL_FAIL_STAGE3)
                yield {"type": "stage3_complete", "data": stage3_result}
            else:
                # Stage 2: Collect rankings
                yield {"type": "stage2_start"}
                stage2_results, label_to_model = await main.stage2_collect_rankings(
                    request.content,
                    stage1_results,
                    models=council_models,
                    zdr_enabled=zdr_enabled,
                    thinking_effort=thinking_effort,
                )
                for index, result in enumerate(stage2_results):
                    for event in main.build_reasoning_stream_events(
                        result,
                        scope="council",
                        stage="stage2",
                        model=result.get("model"),
                        content_key="ranking",
                        index=index,
                    ):
                        yield event
                aggregate_rankings = main.calculate_aggregate_rankings(stage2_results, label_to_model)
                yield {"type": "stage2_complete", "data": stage2_results, "metadata": {"label_to_model": label_to_model, "aggregate_rankings": aggregate_rankings}}

                # Calculate quality metrics for confidence scoring
                from .council import calculate_quality_metrics
                quality_metrics = calculate_quality_metrics(stage2_results, label_to_model)

                # Stage 3: Synthesize final answer with confidence
                yield {"type": "stage3_start"}
                stage3_result = await main.stage3_synthesize_final(
                    request.content,
                    stage1_results,
                    stage2_results,
                    label_to_model,
                    quality_metrics,
                    chairman_model=chairman_model,
                    zdr_enabled=zdr_enabled,
                    thinking_effort=thinking_effort,
                )
                for event in main.build_reasoning_stream_events(
                    stage3_result,
                    scope="council",
                    stage="stage3",
                    model=stage3_result.get("model"),
                    content_key="response",
                ):
                    yield event
                yield {"type": "stage3_complete", "data": stage3_result}

            # Wait for title generation if it was started
            if title_task:
                title = await title_task
                main.storage.update_conversation_title(conversation_id, title)
                yield {"type": "title_complete", "data": {"title": title}}

            # Save complete assistant message with metadata for analytics
            council_metadata = {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings,
                "steward_usage": steward_usage,
                "steward_model": chairman_model or config.CHAIRMAN_MODEL,
            }
            turn_cost = main.calculate_turn_cost(
                mode="council",
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                stage3_result=stage3_result,
                extra_usage_records=extra_usage_records,
            )
            main.storage.add_assistant_message(
                conversation_id,
                stage1_results,
                stage2_results,
                stage3_result,
                council_metadata,  # For analytics tracking
                running_cost=turn_cost,
            )

            # Calculate turn_index BEFORE using it
            updated_conversation = main.storage.get_conversation(conversation_id)
            turn_index = main.get_turn_index(updated_conversation) - 1

            # Index for RAG with enhanced metadata. Skip when the council
            # produced no result: error text must not become a memory. Also
            # skip entirely for ZDR turns (audit §12, Decision #5): PageIndex
            # memory is cross-conversation, so ZDR content must never enter it.
            if stage3_result.get("model") != "error" and not zdr_enabled:
                logger.info("[PHASE1] Indexing turn %d for conversation %s", turn_index, conversation_id)

                # Extract topics from question + final answer
                from .council import extract_topics
                combined_text = request.content + " " + stage3_result.get("response", "")
                topics = await extract_topics(
                    combined_text,
                    max_topics=3,
                    zdr_enabled=zdr_enabled,
                )
                logger.info("[PHASE1] Topics extracted: %s", topics)

                logger.info("[PHASE1] Quality metrics: %s", quality_metrics)

                # Index session with enhanced metadata
                main.rag_system.index_session(
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
                main.rag_system.refresh_hybrid_index()
                logger.info("[PHASE1] Hybrid index refreshed")

        else:
            # Chat mode
            yield {"type": "chat_start"}

            logger.info(f"[CHAT] Chat mode started for query: {request.content[:50]}...")

            # Reload conversation to get history
            updated_conversation = main.storage.get_conversation(conversation_id)
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
            yield {"type": "run_plan", "data": run_plan.to_dict()}

            # PHASE 1: Rewrite query for better RAG retrieval
            from .council import rewrite_query
            logger.info("[CHAT] About to rewrite query...")
            rewritten_query = await rewrite_query(
                request.content,
                updated_conversation["messages"],
                zdr_enabled=zdr_enabled,
            )
            logger.info("[CHAT] Query rewritten, now retrieving RAG context...")

            # Web Search grounding for chat mode (if enabled)
            chat_web_context = ""
            if request.web_search_enabled:
                from .web_search import web_search_stage0
                yield {"type": "web_search_start"}
                search_result = await web_search_stage0(
                    request.content,
                    depth=request.web_search_depth,
                    zdr_enabled=zdr_enabled,
                )
                chat_web_context = search_result.get("context", "")
                if search_result.get("usage"):
                    extra_usage_records.append({
                        "model": search_result.get("model"),
                        "usage": search_result.get("usage", {}),
                    })
                yield {"type": "web_search_complete", "data": {"context": chat_web_context[:500], "citations": search_result.get("citations", []), "model": search_result.get("model", "")}}

            # Retrieve context via PageIndex reasoning RAG
            rag_context = await main.rag_system.retrieve_async(
                rewritten_query,
                conversation_id,
                max_tokens=run_plan.rag_max_tokens,
                zdr_enabled=zdr_enabled,
            )
            logger.info(f"[CHAT] RAG context retrieved ({len(rag_context)} chars), calling chairman...")

            # Chat with chairman (using original query + attachment context)
            effective_chairman_model = (
                run_plan.chairman_model or chairman_model or config.CHAIRMAN_MODEL
            )
            try:
                logger.info(f"[CHAT] Calling chairman with query: {request.content[:50]}...")

                # Combine RAG context with attachment context and web search
                combined_context = rag_context
                if attachment_context:
                    combined_context = f"{attachment_context}\n\n{rag_context}" if rag_context else attachment_context
                if chat_web_context:
                    combined_context = f"[Web Search Results]\n{chat_web_context}\n\n{combined_context}" if combined_context else f"[Web Search Results]\n{chat_web_context}"

                response_dict = await main.chat_with_chairman(
                    request.content,  # Original query to Chairman
                    updated_conversation["messages"],
                    combined_context,
                    chairman_model=effective_chairman_model,
                    zdr_enabled=zdr_enabled,
                    thinking_effort=thinking_effort,
                )
                logger.info("[CHAT] Chairman response received")
            except Exception as e:
                logger.error(f"[CHAT] Error from chairman: {e}")
                response_dict = {
                    "content": f"I apologize, but I encountered an error: {str(e)}",
                    "usage": {}
                }

            for event in main.build_reasoning_stream_events(
                response_dict,
                scope="chat",
                stage="chat",
                model=effective_chairman_model,
                content_key="content",
            ):
                yield event

            # Save chat message
            logger.info("[CHAT] Saving chat message...")
            turn_cost = main.calculate_turn_cost(
                mode="chat",
                response_dict=response_dict,
                chairman_model=effective_chairman_model,
                extra_usage_records=extra_usage_records,
            )
            main.storage.add_chat_message(
                conversation_id,
                response_dict["content"],
                running_cost=turn_cost,
                reasoning=response_dict.get("reasoning"),
            )

            yield {"type": "chat_response", "data": response_dict}
            logger.info("[CHAT] Chat response sent to client")

        # Update conversation cost
        main.storage.update_conversation_cost(conversation_id, turn_cost)

        # Update session usage after current turn cost before checking warnings.
        budget_state = main.storage.record_session_usage(conversation_id, turn_cost)
        warning_level = budget_state["warning_level"]

        # Send budget warning if threshold crossed
        if warning_level is not None:
            warning_pct = int(warning_level * 100)
            logger.info(f"[BUDGET] Emitting warning at {warning_pct}% for conversation {conversation_id}")
            yield {"type": "budget_warning", "data": {"threshold": warning_level, "percentage": warning_pct}}

        # Get updated total cost
        updated_conv = main.storage.get_conversation(conversation_id)
        total_cost = updated_conv.get("total_cost", 0.0)

        # Get budget spent percentage for completion event
        spent_pct = budget_state["budget_spent_pct"]

        # Send completion event with cost info and budget status
        yield {"type": "complete", "data": {"turn_cost": turn_cost, "total_cost": total_cost, "session_usage": budget_state["usage"], "budget_spent_pct": spent_pct}}

    except Exception as e:
        # Send error event
        yield {"type": "error", "message": str(e)}
