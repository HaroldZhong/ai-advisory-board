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

export function formatDuration(durationMs) {
  if (!Number.isFinite(durationMs)) return null;

  if (durationMs < 1000) {
    return `${Math.max(0, Math.round(durationMs))}ms`;
  }

  if (durationMs < 60_000) {
    return `${(durationMs / 1000).toFixed(1).replace(/\.0$/, '')}s`;
  }

  const minutes = Math.floor(durationMs / 60_000);
  const seconds = Math.round((durationMs % 60_000) / 1000);
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
