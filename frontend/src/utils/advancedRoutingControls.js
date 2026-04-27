export const ADVANCED_ROUTING_DISCLOSURE_STORAGE_KEY = 'aab.advancedRouting.expanded';

const RAG_LABELS = {
  auto: 'Auto',
  low: 'Minimal',
  medium: 'Balanced',
  high: 'Extended',
  max: 'Maximum',
};

const MODEL_TIER_LABELS = {
  auto: 'Auto',
  budget: 'Economy',
  mid: 'Balanced',
  premium: 'Premium',
};

const RAG_HINTS = {
  auto: 'Budget-aware context chosen from the message.',
  low: 'Lowest context and lowest routing cost.',
  medium: 'Balanced context and cost.',
  high: 'More context with higher cost.',
  max: '32k context; highest cost and not always better.',
};

const MODEL_TIER_HINTS = {
  auto: 'Uses the selected preset or conversation default.',
  budget: 'Lowest-cost chairman preference.',
  mid: 'Balanced quality and cost.',
  premium: 'Highest-quality chairman preference, highest cost.',
};

function getDefaultStorage() {
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

export function readAdvancedRoutingDisclosurePreference(storage = getDefaultStorage()) {
  try {
    return storage?.getItem(ADVANCED_ROUTING_DISCLOSURE_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

export function writeAdvancedRoutingDisclosurePreference(expanded, storage = getDefaultStorage()) {
  try {
    storage?.setItem(ADVANCED_ROUTING_DISCLOSURE_STORAGE_KEY, expanded ? 'true' : 'false');
  } catch {
    // Non-critical preference; ignore unavailable or blocked storage.
  }
}

export function getRagPresetHint(presetId) {
  return RAG_HINTS[presetId] || RAG_HINTS.auto;
}

export function getModelTierHint(tierId) {
  return MODEL_TIER_HINTS[tierId] || MODEL_TIER_HINTS.auto;
}

export function getAdvancedRoutingSummary(settings = {}) {
  const ragLabel = RAG_LABELS[settings.ragPreset] || RAG_LABELS.auto;
  const tierLabel = MODEL_TIER_LABELS[settings.modelTier] || MODEL_TIER_LABELS.auto;
  return `Context: ${ragLabel} · Model tier: ${tierLabel}`;
}
