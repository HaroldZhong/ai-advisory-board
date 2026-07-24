export const THINKING_EFFORT_LEVELS = ['minimal', 'low', 'medium', 'high', 'xhigh'];

const THINKING_EFFORT_OPTIONS = {
  minimal: {
    label: 'Minimal',
    description: 'Fastest responses with the least hidden reasoning.',
    costHint: '~0.2x reasoning tokens',
    tone: 'neutral',
  },
  low: {
    label: 'Low',
    description: 'Lighter reasoning for simple follow-ups.',
    costHint: '~0.4x reasoning tokens',
    tone: 'neutral',
  },
  medium: {
    label: 'Medium',
    description: 'Balanced reasoning depth for everyday work.',
    costHint: 'Default reasoning token budget',
    tone: 'neutral',
  },
  high: {
    label: 'High',
    description: 'Deeper reasoning for harder analysis and trade-offs.',
    costHint: '~1.6x reasoning tokens',
    tone: 'warn',
  },
  xhigh: {
    label: 'X-High',
    description: 'Maximum reasoning for the hardest questions.',
    costHint: '~1.9x reasoning tokens',
    tone: 'danger',
  },
};

export function isValidThinkingEffort(effort) {
  return THINKING_EFFORT_LEVELS.includes(effort);
}

export function normalizeThinkingEffort(effort, fallback = 'medium') {
  return isValidThinkingEffort(effort) ? effort : fallback;
}

export function getThinkingEffortOption(effort) {
  return THINKING_EFFORT_OPTIONS[normalizeThinkingEffort(effort)] || THINKING_EFFORT_OPTIONS.medium;
}

export function formatThinkingEffortLabel(effort) {
  return getThinkingEffortOption(effort).label;
}

export function getThinkingEffortTone(effort) {
  return getThinkingEffortOption(effort).tone;
}

export function resolveEffectiveThinkingEffort(conversation) {
  // v1.3.0 B3: the preset is a suggestion, never a cap -- return the resolved
  // level as-is (request > stored > preset default > medium).
  const metadata = conversation?.metadata || {};
  if (isValidThinkingEffort(metadata.thinking_effort)) {
    return metadata.thinking_effort;
  }
  if (isValidThinkingEffort(metadata.default_reasoning_effort)) {
    return metadata.default_reasoning_effort;
  }
  return 'medium';
}

export function setConversationThinkingEffortMetadata(currentConversation, conversationId, effort) {
  if (!currentConversation || currentConversation.id !== conversationId) return currentConversation;
  if (effort !== undefined && !isValidThinkingEffort(effort)) return currentConversation;

  const metadata = { ...(currentConversation.metadata || {}) };
  if (effort === undefined) {
    delete metadata.thinking_effort;
  } else {
    metadata.thinking_effort = effort;
  }

  return {
    ...currentConversation,
    metadata,
  };
}

export function mergeConversationThinkingEffortUpdate(currentConversation, updatedConversation) {
  if (!updatedConversation) return currentConversation;
  if (!currentConversation) return currentConversation;
  if (currentConversation.id !== updatedConversation.id) return currentConversation;

  return {
    ...currentConversation,
    metadata: {
      ...(currentConversation.metadata || {}),
      ...(updatedConversation.metadata || {}),
    },
  };
}
