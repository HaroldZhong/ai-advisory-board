const COUNCIL_STAGES = new Set(['stage1', 'stage2', 'stage3']);

/**
 * Transient stream reasoning is held on the active assistant message until the
 * completed stage payload arrives. Council buffers are keyed by index first,
 * then model id, then a default slot for singleton stages like stage3.
 */
function appendText(existing, next) {
  return `${existing || ''}${next || ''}`;
}

function getReasoningSlot(data) {
  if (data?.index !== undefined && data?.index !== null) {
    return String(data.index);
  }
  return data?.model || 'default';
}

export function appendReasoningDeltaToMessage(message, data) {
  if (!message || !data?.text) return message;

  if (data.scope === 'chat' || data.stage === 'chat') {
    return {
      ...message,
      reasoning: appendText(message.reasoning, data.text),
    };
  }

  if (!COUNCIL_STAGES.has(data.stage)) {
    console.warn('Unknown reasoning stream stage:', data.stage);
    return message;
  }

  const reasoningBuffers = { ...(message.reasoningBuffers || {}) };
  const stageBuffers = { ...(reasoningBuffers[data.stage] || {}) };
  const slot = getReasoningSlot(data);
  const existing = stageBuffers[slot] || {};

  stageBuffers[slot] = {
    ...existing,
    index: data.index,
    model: data.model,
    text: appendText(existing.text, data.text),
  };

  reasoningBuffers[data.stage] = stageBuffers;

  return {
    ...message,
    reasoningBuffers,
  };
}

export function appendContentDeltaToMessage(message, data) {
  // Council content deltas are intentionally ignored here; completed stage
  // payloads remain the source of truth for Stage 1/2/3 visible content.
  if (!message || !data?.text || data.stage !== 'chat') return message;

  return {
    ...message,
    content: appendText(message.content, data.text),
  };
}

export function applyStreamUpdateToActiveConversation(conversation, targetConversationId, updater) {
  if (!conversation || conversation.id !== targetConversationId) {
    return conversation;
  }
  return updater(conversation);
}

export function markLastAssistantStreamInterrupted(conversation) {
  if (!conversation?.messages?.length) return conversation;

  const messages = [...conversation.messages];
  const lastIndex = messages.length - 1;
  const lastMessage = messages[lastIndex];
  if (lastMessage?.role !== 'assistant' || !lastMessage.loading) {
    return conversation;
  }

  const nextLoading = { ...lastMessage.loading };
  for (const key of Object.keys(nextLoading)) {
    if (typeof nextLoading[key] === 'boolean') {
      nextLoading[key] = false;
    }
  }
  if ('stage3_status' in nextLoading) {
    nextLoading.stage3_status = 'error';
  }

  messages[lastIndex] = {
    ...lastMessage,
    loading: nextLoading,
  };

  return {
    ...conversation,
    messages,
  };
}

export function mergeReasoningBuffersIntoResults(results, buffers = {}) {
  if (!Array.isArray(results)) return results;

  return results.map((result, index) => {
    if (result?.reasoning) return result;

    const indexedBuffer = buffers[String(index)];
    const modelBuffer = result?.model ? buffers[result.model] : null;
    const buffer = indexedBuffer || modelBuffer;
    if (!buffer?.text) return result;

    return {
      ...result,
      reasoning: buffer.text,
    };
  });
}

export function mergeReasoningBufferIntoResult(result, buffers = {}) {
  if (!result || result.reasoning) return result;

  const modelBuffer = result.model ? buffers[result.model] : null;
  const firstBuffer = Object.values(buffers)[0];
  const buffer = modelBuffer || firstBuffer;
  if (!buffer?.text) return result;

  return {
    ...result,
    reasoning: buffer.text,
  };
}
