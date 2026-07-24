export function hasReasoningText(reasoningText) {
  return typeof reasoningText === 'string' && reasoningText.trim().length > 0;
}

export function formatTokenCount(tokenCount) {
  if (!Number.isFinite(tokenCount)) return null;

  if (tokenCount < 1000) {
    return `${tokenCount} tokens`;
  }

  if (tokenCount < 1_000_000) {
    return `${(tokenCount / 1000).toFixed(1)}k tokens`;
  }

  return `${(tokenCount / 1_000_000).toFixed(1)}M tokens`;
}

// v1.3.0 B5/E3 (§3d): the honest post-turn reasoning actuals for one member. Keyed
// ONLY on the reasoning_tokens COUNT (from B5), never on reasoning text -- so 0 or
// absent tokens read as "not available" even when reasoning text was returned, and
// this never implies "no reasoning happened" from empty text.
export function formatReasoningActuals(reasoningTokens) {
  if (Number.isFinite(reasoningTokens) && reasoningTokens > 0) {
    return `reasoning: ${formatTokenCount(reasoningTokens)}`;
  }
  return 'reasoning: not available';
}

export function formatDuration(durationMs) {
  if (!Number.isFinite(durationMs)) return null;

  if (durationMs < 1000) {
    return `${Math.max(0, Math.round(durationMs))}ms`;
  }

  const roundedSeconds = Math.round(Math.max(0, durationMs) / 1000);

  if (roundedSeconds < 60) {
    return `${(durationMs / 1000).toFixed(1).replace(/\.0$/, '')}s`;
  }

  const minutes = Math.floor(roundedSeconds / 60);
  const seconds = roundedSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

export function getReasoningStatusLabel(status) {
  switch (status) {
    case 'complete':
      return 'Reasoning complete';
    case 'unavailable':
      return 'Reasoning unavailable';
    case 'streaming':
      return 'Reasoning';
    default:
      return 'Reasoning complete';
  }
}
