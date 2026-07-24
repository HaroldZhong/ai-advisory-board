import {
  getThinkingEffortOption,
  resolveEffectiveThinkingEffort,
} from './thinkingEffort.js';

export function formatCurrency(value) {
  const amount = Number(value || 0);
  if (amount > 0 && amount < 0.01) return `$${amount.toFixed(4)}`;
  return `$${amount.toFixed(2)}`;
}

// v1.3.0 D2: single-source the budget notify tiers from the served policy's
// `notify_thresholds` (config.py SESSION_POLICY_DEFAULTS) instead of triplicating
// the constants here / in SessionBudgetSelector. Fall back to the historical
// [caution, warn, danger] only when a policy omits them.
export const DEFAULT_NOTIFY_THRESHOLDS = [0.75, 0.85, 1.0];

// "Budget reached" is the actual hard cap (spentPct >= 1.0), independent of the
// served notify tiers -- the backend only 409s at 1.0, so the UI must not claim
// "reached" at a lower notify tier (or fail to at 100% when a tier exceeds 1.0).
const CAP_REACHED = 1.0;

// Accept ANY backend-valid list: non-empty, positive, finite numbers. The backend
// validates and emits events for any sorted list (2, 3, 4… tiers), so the UI must
// mirror whatever it served rather than only a 3-tier shape.
export function resolveNotifyThresholds(policy) {
  const served = policy?.notify_thresholds;
  if (Array.isArray(served) && served.length > 0
      && served.every((n) => typeof n === 'number' && Number.isFinite(n) && n > 0)) {
    return [...served].sort((a, b) => a - b);
  }
  return DEFAULT_NOTIFY_THRESHOLDS;
}

// Warning tiers are the served thresholds strictly below the hard cap.
function subCapTiers(thresholds) {
  return thresholds.filter((t) => t < CAP_REACHED);
}

const WARN_TEXT = {
  body: 'Premium routing may be reduced on upcoming turns.',
  action: 'Raise cap',
};
const CAUTION_TEXT = {
  body: 'You are approaching this conversation budget.',
  action: 'Adjust',
};
const REACHED_TEXT = {
  level: 'danger',
  label: 'Budget reached',
  body: 'Raise the cap to keep spending predictable.',
  action: 'Raise cap',
};

export function getBudgetTone(spentPct, thresholds = DEFAULT_NOTIFY_THRESHOLDS) {
  if (spentPct == null) return 'neutral';
  if (spentPct >= CAP_REACHED) return 'danger';
  const sub = subCapTiers(thresholds);
  if (sub.length && spentPct >= sub[sub.length - 1]) return 'warn';
  if (sub.length && spentPct >= sub[0]) return 'caution';
  return 'neutral';
}

export function getBudgetWarningText(spentPct, thresholds = DEFAULT_NOTIFY_THRESHOLDS) {
  if (spentPct == null) return null;
  if (spentPct >= CAP_REACHED) return REACHED_TEXT;
  const sub = subCapTiers(thresholds);
  const crossed = sub.filter((t) => spentPct >= t);
  if (crossed.length === 0) return null;
  const highestCrossed = crossed[crossed.length - 1];
  const isWarn = highestCrossed === sub[sub.length - 1];
  return {
    level: isWarn ? 'warn' : 'caution',
    label: `${Math.round(highestCrossed * 100)}% used`,
    ...(isWarn ? WARN_TEXT : CAUTION_TEXT),
  };
}

function getBudgetWarningTextForThreshold(threshold, thresholds = DEFAULT_NOTIFY_THRESHOLDS) {
  if (threshold == null) return null;
  if (threshold >= CAP_REACHED) return REACHED_TEXT;
  const sub = subCapTiers(thresholds);
  const isWarn = sub.length > 0 && threshold >= sub[sub.length - 1];
  return {
    level: isWarn ? 'warn' : 'caution',
    label: `${Math.round(threshold * 100)}% used`,
    ...(isWarn ? WARN_TEXT : CAUTION_TEXT),
  };
}

export function getEffectiveBudgetWarning(spentPct, eventThreshold, thresholds = DEFAULT_NOTIFY_THRESHOLDS) {
  const currentWarning = getBudgetWarningText(spentPct, thresholds);
  if (currentWarning) return currentWarning;
  if (eventThreshold == null) return null;
  if (spentPct != null && spentPct < eventThreshold) return null;
  return getBudgetWarningTextForThreshold(eventThreshold, thresholds);
}

export function getBudgetCapBlockState(conversation) {
  const policy = conversation?.session_policy || {};
  const usage = conversation?.session_usage || {};
  const budgetUsd = policy.budget_usd ?? null;
  const spentPct = getSpentPct(conversation, policy, usage);
  const allowOverage = policy.allow_overage !== false;
  const blocked = budgetUsd != null && !allowOverage && spentPct != null && spentPct >= 1;

  if (!blocked) {
    return { blocked: false };
  }

  return {
    blocked: true,
    label: 'Budget reached',
    detail: 'Raise the cap before sending another message.',
    action: 'Raise cap',
  };
}

// v1.3.0 D3: shape the session-policy update from the budget modal, carrying the
// hard-cap opt-in. A hard cap is meaningless without a limit, so "No Limit"
// always allows overage; otherwise the user's choice is honored.
export function buildBudgetPolicyUpdate(budgetUsd, allowOverage = true) {
  const noLimit = budgetUsd === null || budgetUsd === undefined;
  return {
    budget_usd: budgetUsd ?? null,
    allow_overage: noLimit ? true : Boolean(allowOverage),
  };
}

export function resolveEffectiveZdr(conversation, settings = {}, zdrAvailable = true) {
  const metadataZdr = conversation?.metadata?.zdr_enabled;
  // An explicit conversation-level ZDR choice is the privacy promise made at
  // creation time — it must still surface (and be rejectable by the backend)
  // even off-OpenRouter, so the user consciously turns it off rather than
  // having it silently downgraded.
  if (metadataZdr === true || metadataZdr === false) return metadataZdr;
  // A bare global default preference is not a promise to anyone — don't let
  // it brick sends on a provider that can't do ZDR at all.
  return zdrAvailable && settings.zdrEnabled === true;
}

export function resolveAttachmentEnhancementZdr(effectiveZdr, settings = {}) {
  if (effectiveZdr === true || effectiveZdr === false) return effectiveZdr;
  return settings.zdrEnabled === true;
}

export function mergeConversationPrivacyUpdate(currentConversation, updatedConversation) {
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

export function setConversationPrivacyMetadata(currentConversation, conversationId, zdrEnabled) {
  if (!currentConversation || currentConversation.id !== conversationId) return currentConversation;

  const metadata = { ...(currentConversation.metadata || {}) };
  if (zdrEnabled === true || zdrEnabled === false) {
    metadata.zdr_enabled = zdrEnabled;
  } else {
    delete metadata.zdr_enabled;
  }

  return {
    ...currentConversation,
    metadata,
  };
}

export function getPrivacyToggleDisabledReason({
  isStreaming = false,
  isUploading = false,
  isUpdatingPrivacy = false,
  isEnablingUnavailable = false,
} = {}) {
  // Deliberate partial revert of P3-T8 item 2 (Codex review, round 4):
  // prepare_turn resolves zdr_enabled ONCE per turn, so an in-flight turn
  // keeps its captured routing regardless of what the metadata says by the
  // time it finishes. Memory writes are safe either way (the #71 ZDR write
  // barrier), but flipping the toggle mid-stream would show "ZDR enforced"
  // while the CURRENT turn is still routed under the old setting — a
  // per-turn privacy promise the UI must not appear to break. Thinking
  // effort has no such promise (it only applies to future turns), so it
  // stays enabled mid-stream.
  if (isStreaming) return 'Wait for the current response to finish';
  if (isUploading) return 'Privacy changes are disabled while attachments are uploading';
  if (isUpdatingPrivacy) return 'Privacy update is being saved';
  if (isEnablingUnavailable) return 'Requires OpenRouter';
  return null;
}

// E3: the preset label is single-sourced from the served preset (stored in the
// conversation metadata at creation), not a frontend hardcode. Pre-label
// conversations fall back to a label derived from the id.
function formatPresetLabel(presetId, presetLabel) {
  if (presetLabel) return presetLabel;
  if (!presetId) return 'Custom';
  return presetId.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function getSpentPct(conversation, policy, usage) {
  if (conversation?.budget_spent_pct != null) return conversation.budget_spent_pct;
  const budgetUsd = policy.budget_usd;
  if (budgetUsd == null || budgetUsd <= 0) return null;
  return Number(usage.spent_usd || 0) / budgetUsd;
}

export function formatTrustRowState({
  conversation,
  settings = {},
  attachmentCount = 0,
  zdrAvailable = true,
} = {}) {
  const metadata = conversation?.metadata || {};
  const policy = conversation?.session_policy || {};
  const usage = conversation?.session_usage || {};
  const council = metadata.council_models || [];
  const effectiveZdr = resolveEffectiveZdr(conversation, settings, zdrAvailable);
  const spentUsd = Number(usage.spent_usd || 0);
  const budgetUsd = policy.budget_usd ?? null;
  const spentPct = getSpentPct(conversation, policy, usage);
  const notifyThresholds = resolveNotifyThresholds(policy);
  const budgetCapBlock = getBudgetCapBlockState(conversation);
  const pctLabel = spentPct == null ? null : `${Math.round(spentPct * 100)}% used`;
  const webEnabled = settings.webSearchEnabled === true;
  const webDepth = settings.webSearchDepth || 'fast';
  const effectiveThinkingEffort = resolveEffectiveThinkingEffort(conversation);
  const thinkingOption = getThinkingEffortOption(effectiveThinkingEffort);

  const isChatDefault = metadata.default_mode === 'chat';

  return {
    council: isChatDefault
      ? {
        label: 'Chat',
        detail: metadata.chairman_model || 'Default model',
        count: 0,
        presetId: null,
      }
      : {
        label: formatPresetLabel(metadata.preset_id, metadata.preset_label),
        detail: `${council.length || 0} council + 1 chair`,
        count: council.length || 0,
        presetId: metadata.preset_id || null,
      },
    privacy: {
      effectiveZdr,
      label: effectiveZdr ? 'ZDR enforced' : 'Standard',
      detail: effectiveZdr ? 'Zero data retention routes' : 'Provider retention may apply',
      locked: metadata.preset_id === 'private',
    },
    budget: {
      budgetUsd,
      spentUsd,
      spentPct,
      tone: getBudgetTone(spentPct, notifyThresholds),
      label: budgetUsd == null
        ? 'No budget'
        : `${formatCurrency(spentUsd)} / ${formatCurrency(budgetUsd)}`,
      detail: budgetUsd == null ? 'Set session limit' : pctLabel,
      warning: getBudgetWarningText(spentPct, notifyThresholds),
      notifyThresholds,
      capBlock: budgetCapBlock,
    },
    tools: {
      webEnabled,
      webDepth,
      attachmentCount,
      label: webEnabled ? `Web ${webDepth}` : 'Web off',
      detail: attachmentCount > 0
        ? `${attachmentCount} ${attachmentCount === 1 ? 'file' : 'files'} attached`
        : 'No files attached',
    },
    thinking: {
      value: effectiveThinkingEffort,
      label: thinkingOption.label,
      detail: thinkingOption.costHint,
      description: thinkingOption.description,
      tone: thinkingOption.tone,
    },
    cost: {
      value: formatCurrency(conversation?.total_cost || 0),
      label: 'Session cost',
    },
  };
}
