const COUNCIL_STAGES = new Set(['stage1', 'stage2', 'stage3']);

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
  if (!message || !data?.text || data.stage !== 'chat') return message;

  return {
    ...message,
    content: appendText(message.content, data.text),
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
