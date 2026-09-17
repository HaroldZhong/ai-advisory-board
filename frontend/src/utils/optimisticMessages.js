import { markLastAssistantStreamInterrupted } from './reasoningMessages.js';

// Reconcile only this run, never replace the whole conversation with a late GET.
export function reconcileInterruptedRun(conversation, persisted, { conversationId, requestId, runId, reason }) {
  if (conversation?.id !== conversationId) return conversation;
  const index = conversation.messages.findIndex((m) => m.client_request_id === requestId);
  if (index < 0) return conversation;
  const saved = persisted?.id === conversationId && runId
    ? persisted.messages.find((m) => m.role === 'assistant' && m.metadata?.run_id === runId)
    : null;
  const visible = conversation.messages[index];
  const interrupted = markLastAssistantStreamInterrupted({ messages: [visible] }).messages[0];
  const messages = [...conversation.messages];
  messages[index] = saved
    ? { ...saved, client_request_id: requestId, interruption: reason }
    : { ...interrupted, interruption: reason, unconfirmed: true };
  return { ...conversation, messages };
}

export function rollbackFailedSendMessages(
  messages,
  { editIndex = -1, previousMessages = null } = {},
) {
  if (editIndex >= 0 && Array.isArray(previousMessages)) {
    return [...previousMessages];
  }

  const nextMessages = [...(messages || [])];
  if (nextMessages.length >= 2 && nextMessages[nextMessages.length - 1]?.role === 'assistant') {
    nextMessages.splice(-2);
  } else if (nextMessages.length >= 1 && nextMessages[nextMessages.length - 1]?.role === 'user') {
    nextMessages.splice(-1);
  }
  return nextMessages;
}

export function rollbackFailedSendConversation(
  conversation,
  { conversationId, editIndex = -1, previousMessages = null } = {},
) {
  if (!conversation || conversation.id !== conversationId) {
    return conversation;
  }

  return {
    ...conversation,
    messages: rollbackFailedSendMessages(conversation.messages, {
      editIndex,
      previousMessages,
    }),
  };
}
