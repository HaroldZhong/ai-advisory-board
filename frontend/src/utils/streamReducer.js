/**
 * Pure reducer for SSE turn events (backend/turn_pipeline.py, audit §9).
 *
 * Moves the per-event conversation-update logic that used to live inline in
 * App.jsx's handleSendMessage callback into a single, testable function.
 * Event shapes mirror what turn_pipeline.py yields; unhandled event types
 * (e.g. edit_truncated, web_search_*, steward_*, run_plan) are intentionally
 * no-ops here, matching App.jsx's prior `default: console.warn(...)` branch.
 *
 * State shape:
 *   {
 *     conversation: <the active conversation object, or null>,
 *     isLoading: boolean,
 *     budgetWarning: object | null,
 *   }
 *
 * `context.availableModels` is required for the stage cost calculations,
 * mirroring the `availableModels` closure App.jsx's switch used to read.
 */
import {
  appendContentDeltaToMessage,
  appendReasoningDeltaToMessage,
  markLastAssistantStreamInterrupted,
  mergeReasoningBufferIntoResult,
  mergeReasoningBuffersIntoResults,
} from './reasoningMessages.js';
import {
  calculateStage1Cost,
  calculateStage2Cost,
  calculateStage3Cost,
} from './cost.js';

function updateConversation(state, updater) {
  if (!state.conversation) return state;
  return { ...state, conversation: updater(state.conversation) };
}

function updateLastMessage(state, updater) {
  return updateConversation(state, (conversation) => {
    const messages = [...conversation.messages];
    const lastIndex = messages.length - 1;
    if (lastIndex < 0) return conversation;
    messages[lastIndex] = updater(messages[lastIndex]);
    return { ...conversation, messages };
  });
}

export function streamReducer(state, event, context = {}) {
  const { availableModels } = context;

  switch (event.type) {
    case 'stage1_start':
      return updateLastMessage(state, (msg) => ({
        ...msg,
        loading: { ...msg.loading, stage1: true },
      }));

    case 'stage1_complete':
      return updateLastMessage(state, (msg) => {
        const stage1 = mergeReasoningBuffersIntoResults(
          event.data,
          msg.reasoningBuffers?.stage1,
        );
        const cost = calculateStage1Cost(stage1, availableModels);
        return {
          ...msg,
          stage1,
          loading: { ...msg.loading, stage1: false },
          running_cost: (msg.running_cost || 0) + cost,
        };
      });

    case 'stage2_start':
      return updateLastMessage(state, (msg) => ({
        ...msg,
        loading: { ...msg.loading, stage2: true },
      }));

    case 'stage2_complete':
      return updateLastMessage(state, (msg) => {
        const stage2 = mergeReasoningBuffersIntoResults(
          event.data,
          msg.reasoningBuffers?.stage2,
        );
        const cost = calculateStage2Cost(stage2, availableModels);
        return {
          ...msg,
          stage2,
          metadata: event.metadata,
          loading: { ...msg.loading, stage2: false },
          running_cost: (msg.running_cost || 0) + cost,
        };
      });

    case 'stage3_start':
      return updateLastMessage(state, (msg) => ({
        ...msg,
        loading: { ...msg.loading, stage3: true },
      }));

    case 'stage3_complete':
      return updateLastMessage(state, (msg) => {
        const stage3 = mergeReasoningBufferIntoResult(
          event.data,
          msg.reasoningBuffers?.stage3,
        );
        const cost = calculateStage3Cost(stage3, availableModels);
        return {
          ...msg,
          stage3,
          loading: { ...msg.loading, stage3: false },
          running_cost: (msg.running_cost || 0) + cost,
        };
      });

    case 'chat_start':
      return updateLastMessage(state, (msg) => ({
        ...msg,
        loading: { ...msg.loading, chat: true },
      }));

    case 'chat_response':
      return updateLastMessage(state, (msg) => {
        const next = { ...msg, loading: { ...msg.loading, chat: false } };
        if (typeof event.data === 'string') {
          next.content = event.data;
        } else {
          next.content = event.data.content;
          if (event.data.reasoning) {
            next.reasoning = event.data.reasoning;
          }
        }
        return next;
      });

    case 'reasoning_delta':
      return updateLastMessage(state, (msg) => appendReasoningDeltaToMessage(msg, event.data));

    case 'content_delta':
      return updateLastMessage(state, (msg) => appendContentDeltaToMessage(msg, event.data));

    case 'title_complete':
      // Side effect (reloading the conversation list) is owned by the hook.
      return state;

    case 'budget_warning':
      return { ...state, budgetWarning: event.data };

    case 'complete':
      return {
        ...updateConversation(state, (conversation) => {
          if (!event.data) return conversation;
          const messages = [...conversation.messages];
          const lastIndex = messages.length - 1;
          const lastMsg = messages[lastIndex];
          if (lastMsg?.role === 'assistant' && event.data.turn_cost != null) {
            messages[lastIndex] = { ...lastMsg, running_cost: event.data.turn_cost };
          }
          return {
            ...conversation,
            messages,
            total_cost: event.data.total_cost,
            session_usage: event.data.session_usage,
            budget_spent_pct: event.data.budget_spent_pct,
          };
        }),
        isLoading: false,
      };

    case 'error':
      return {
        ...updateConversation(state, markLastAssistantStreamInterrupted),
        isLoading: false,
      };

    default:
      return state;
  }
}
