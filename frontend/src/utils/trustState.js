import {
  getThinkingEffortOption,
  resolveEffectiveThinkingEffort,
} from './thinkingEffort.js';

const PRESET_LABELS = {
  balanced: 'Balanced',
  research: 'Research',
  budget: 'Budget',
  private: 'Private',
};

export function formatCurrency(value) {
  const amount = Number(value || 0);
  if (amount > 0 && amount < 0.01) return `$${amount.toFixed(4)}`;
  return `$${amount.toFixed(2)}`;
}

export function getBudgetTone(spentPct) {
  if (spentPct == null) return 'neutral';
  if (spentPct >= 1) return 'danger';
  if (spentPct >= 0.85) return 'warn';
  if (spentPct >= 0.75) return 'caution';
  return 'neutral';
}

export function getBudgetWarningText(spentPct) {
  if (spentPct == null || spentPct < 0.75) return null;
  if (spentPct >= 1) {
    return {
      level: 'danger',
      label: 'Budget reached',
      body: 'Raise the cap to keep spending predictable.',
      action: 'Raise cap',
    };
  }
  if (spentPct >= 0.85) {
    return {
      level: 'warn',
      label: '85% used',
      body: 'Premium routing may be reduced on upcoming turns.',
      action: 'Raise cap',
    };
  }
  return {
    level: 'caution',
    label: '75% used',
    body: 'You are approaching this conversation budget.',
    action: 'Adjust',
  };
}

function getBudgetWarningTextForThreshold(threshold) {
  if (threshold == null) return null;
  if (threshold >= 1) {
    return {
      level: 'danger',
      label: 'Budget reached',
      body: 'Raise the cap to keep spending predictable.',
      action: 'Raise cap',
    };
  }
  if (threshold >= 0.85) {
    return {
      level: 'warn',
      label: '85% used',
      body: 'Premium routing may be reduced on upcoming turns.',
      action: 'Raise cap',
    };
  }
  return {
    level: 'caution',
    label: `${Math.round(threshold * 100)}% used`,
    body: 'You are approaching this conversation budget.',
    action: 'Adjust',
  };
}

export function getEffectiveBudgetWarning(spentPct, eventThreshold) {
  const currentWarning = getBudgetWarningText(spentPct);
  if (currentWarning) return currentWarning;
  if (eventThreshold == null) return null;
  if (spentPct != null && spentPct < eventThreshold) return null;
  return getBudgetWarningTextForThreshold(eventThreshold);
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

export function resolveEffectiveZdr(conversation, settings = {}) {
  const metadataZdr = conversation?.metadata?.zdr_enabled;
  if (metadataZdr === true || metadataZdr === false) return metadataZdr;
  return settings.zdrEnabled === true;
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
  isLoading = false,
  isUploading = false,
  isUpdatingPrivacy = false,
} = {}) {
  if (isLoading) return 'Privacy changes are disabled while a response is streaming';
  if (isUploading) return 'Privacy changes are disabled while attachments are uploading';
  if (isUpdatingPrivacy) return 'Privacy update is being saved';
  return null;
}

function formatPresetLabel(presetId) {
  if (!presetId) return 'Custom';
  return PRESET_LABELS[presetId] || presetId.replace(/[-_]/g, ' ');
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
} = {}) {
  const metadata = conversation?.metadata || {};
  const policy = conversation?.session_policy || {};
  const usage = conversation?.session_usage || {};
  const council = metadata.council_models || [];
  const effectiveZdr = resolveEffectiveZdr(conversation, settings);
  const spentUsd = Number(usage.spent_usd || 0);
  const budgetUsd = policy.budget_usd ?? null;
  const spentPct = getSpentPct(conversation, policy, usage);
  const budgetCapBlock = getBudgetCapBlockState(conversation);
  const pctLabel = spentPct == null ? null : `${Math.round(spentPct * 100)}% used`;
  const webEnabled = settings.webSearchEnabled === true;
  const webDepth = settings.webSearchDepth || 'fast';
  const effectiveThinkingEffort = resolveEffectiveThinkingEffort(conversation);
  const thinkingOption = getThinkingEffortOption(effectiveThinkingEffort);

  return {
    council: {
      label: formatPresetLabel(metadata.preset_id),
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
      tone: getBudgetTone(spentPct),
      label: budgetUsd == null
        ? 'No budget'
        : `${formatCurrency(spentUsd)} / ${formatCurrency(budgetUsd)}`,
      detail: budgetUsd == null ? 'Set session limit' : pctLabel,
      warning: getBudgetWarningText(spentPct),
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
