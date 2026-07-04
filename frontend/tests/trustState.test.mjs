import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatCurrency,
  getEffectiveBudgetWarning,
  formatTrustRowState,
  getBudgetTone,
  getBudgetWarningText,
  getBudgetCapBlockState,
  getPrivacyToggleDisabledReason,
  mergeConversationPrivacyUpdate,
  resolveAttachmentEnhancementZdr,
  resolveEffectiveZdr,
  setConversationPrivacyMetadata,
} from '../src/utils/trustState.js';

test('budget tone follows 75, 85, and 100 percent thresholds', () => {
  assert.equal(getBudgetTone(null), 'neutral');
  assert.equal(getBudgetTone(0.74), 'neutral');
  assert.equal(getBudgetTone(0.75), 'caution');
  assert.equal(getBudgetTone(0.85), 'warn');
  assert.equal(getBudgetTone(1), 'danger');
  assert.equal(getBudgetTone(1.2), 'danger');
});

test('budget warning text includes non-color signals at each threshold', () => {
  assert.equal(getBudgetWarningText(0.74), null);

  assert.deepEqual(getBudgetWarningText(0.75), {
    level: 'caution',
    label: '75% used',
    body: 'You are approaching this conversation budget.',
    action: 'Adjust',
  });

  assert.deepEqual(getBudgetWarningText(0.85), {
    level: 'warn',
    label: '85% used',
    body: 'Premium routing may be reduced on upcoming turns.',
    action: 'Raise cap',
  });

  assert.deepEqual(getBudgetWarningText(1), {
    level: 'danger',
    label: 'Budget reached',
    body: 'Raise the cap to keep spending predictable.',
    action: 'Raise cap',
  });
});

test('budget cap block state only blocks enforced budgets at or above cap', () => {
  assert.deepEqual(
    getBudgetCapBlockState({
      session_policy: { budget_usd: 1, allow_overage: false },
      session_usage: { spent_usd: 1 },
    }),
    {
      blocked: true,
      label: 'Budget reached',
      detail: 'Raise the cap before sending another message.',
      action: 'Raise cap',
    },
  );

  assert.equal(
    getBudgetCapBlockState({
      session_policy: { budget_usd: 1, allow_overage: true },
      session_usage: { spent_usd: 1.5 },
    }).blocked,
    false,
  );

  assert.equal(
    getBudgetCapBlockState({
      session_policy: { budget_usd: 1, allow_overage: false },
      session_usage: { spent_usd: 0.99 },
    }).blocked,
    false,
  );

  assert.equal(getBudgetCapBlockState({ session_policy: {}, session_usage: {} }).blocked, false);
});

test('event warning fallback is ignored after live spend drops below threshold', () => {
  assert.equal(getEffectiveBudgetWarning(0.4, 0.85), null);
  assert.equal(getEffectiveBudgetWarning(0.74, 0.75), null);

  assert.equal(getEffectiveBudgetWarning(null, 0.85).label, '85% used');
  assert.equal(getEffectiveBudgetWarning(0.86, 0.75).label, '85% used');
});

test('legacy event warning fallback supports 70 percent thresholds', () => {
  assert.equal(getBudgetWarningText(0.70), null);
  assert.equal(getEffectiveBudgetWarning(null, 0.70).label, '70% used');
  assert.equal(getEffectiveBudgetWarning(0.70, 0.70).label, '70% used');
  assert.equal(getEffectiveBudgetWarning(0.69, 0.70), null);
});

test('trust row state prefers conversation ZDR metadata over global settings', () => {
  const state = formatTrustRowState({
    conversation: {
      metadata: {
        zdr_enabled: false,
        preset_id: 'balanced',
        council_models: ['a', 'b', 'c', 'd', 'e'],
        chairman_model: 'anthropic/claude-opus-4.7',
      },
      session_policy: { budget_usd: 2 },
      session_usage: { spent_usd: 1 },
      budget_spent_pct: 0.5,
      total_cost: 1.25,
    },
    settings: { zdrEnabled: true, webSearchEnabled: true, webSearchDepth: 'deep' },
    attachmentCount: 2,
  });

  assert.equal(state.privacy.effectiveZdr, false);
  assert.equal(state.privacy.label, 'Standard');
  assert.equal(state.council.label, 'Balanced');
  assert.equal(state.council.detail, '5 council + 1 chair');
  assert.equal(state.budget.tone, 'neutral');
  assert.equal(state.tools.label, 'Web deep');
  assert.equal(state.tools.detail, '2 files attached');
  assert.equal(state.cost.value, '$1.25');
});

test('legacy conversation without ZDR metadata falls back to current settings', () => {
  const state = formatTrustRowState({
    conversation: {
      metadata: { council_models: ['a'], chairman_model: 'chair', default_reasoning_effort: 'high' },
      session_policy: {},
      session_usage: {},
    },
    settings: { zdrEnabled: true, webSearchEnabled: false },
  });

  assert.equal(state.privacy.effectiveZdr, true);
  assert.equal(state.privacy.label, 'ZDR enforced');
  assert.equal(state.thinking.label, 'High');
  assert.equal(state.thinking.detail, '~1.6x reasoning tokens');
  assert.equal(state.thinking.tone, 'warn');
});

test('trust row state ignores the settings ZDR default off-provider', () => {
  const state = formatTrustRowState({
    conversation: {
      metadata: { council_models: ['a'], chairman_model: 'chair' },
      session_policy: {},
      session_usage: {},
    },
    settings: { zdrEnabled: true },
    zdrAvailable: false,
  });

  assert.equal(state.privacy.effectiveZdr, false);
  assert.equal(state.privacy.label, 'Standard');
});

test('chat-default conversations show the active model instead of council preset info', () => {
  const state = formatTrustRowState({
    conversation: {
      metadata: {
        default_mode: 'chat',
        chairman_model: 'anthropic/claude-opus-4.7',
      },
      session_policy: {},
      session_usage: {},
    },
    settings: {},
  });

  assert.equal(state.council.label, 'Chat');
  assert.equal(state.council.detail, 'anthropic/claude-opus-4.7');
  assert.equal(state.council.count, 0);
});

test('privacy update merge preserves local streaming messages', () => {
  const current = {
    id: 'conversation-1',
    messages: [
      { role: 'user', content: 'Question' },
      { role: 'assistant', content: '', loading: { chat: true } },
    ],
    metadata: { preset_id: 'balanced', zdr_enabled: false },
  };
  const serverSnapshot = {
    id: 'conversation-1',
    messages: [],
    metadata: { preset_id: 'balanced', zdr_enabled: true },
  };

  const merged = mergeConversationPrivacyUpdate(current, serverSnapshot);

  assert.equal(merged.messages, current.messages);
  assert.equal(merged.metadata.zdr_enabled, true);
});

test('privacy update merge ignores stale responses for inactive conversations', () => {
  const current = {
    id: 'conversation-2',
    messages: [{ role: 'user', content: 'Active conversation' }],
    metadata: { preset_id: 'balanced', zdr_enabled: false },
  };
  const staleServerSnapshot = {
    id: 'conversation-1',
    messages: [{ role: 'user', content: 'Old conversation' }],
    metadata: { preset_id: 'balanced', zdr_enabled: true },
  };

  assert.equal(
    mergeConversationPrivacyUpdate(current, staleServerSnapshot),
    current,
  );
  assert.equal(mergeConversationPrivacyUpdate(null, staleServerSnapshot), null);
});

test('local privacy metadata update only applies to the active conversation', () => {
  const current = {
    id: 'conversation-1',
    messages: [],
    metadata: { preset_id: 'balanced', zdr_enabled: false },
  };

  const updated = setConversationPrivacyMetadata(current, 'conversation-1', true);
  assert.notEqual(updated, current);
  assert.equal(updated.metadata.zdr_enabled, true);
  assert.equal(updated.metadata.preset_id, 'balanced');

  assert.equal(
    setConversationPrivacyMetadata(current, 'conversation-2', true),
    current,
  );

  const restored = setConversationPrivacyMetadata(updated, 'conversation-1', undefined);
  assert.equal(restored.metadata.zdr_enabled, undefined);
  assert.equal(restored.metadata.preset_id, 'balanced');
});

test('privacy toggle is disabled for active data-routing operations', () => {
  assert.equal(getPrivacyToggleDisabledReason(), null);
  assert.match(
    getPrivacyToggleDisabledReason({ isUploading: true }),
    /attachments are uploading/,
  );
  assert.match(
    getPrivacyToggleDisabledReason({ isUpdatingPrivacy: true }),
    /being saved/,
  );
});

test('privacy toggle is blocked while a turn is streaming (Codex review, P3-T8 round 4)', () => {
  // Deliberate partial revert of round 2's "usable mid-stream" change:
  // prepare_turn resolves zdr_enabled ONCE per turn, so flipping this
  // mid-stream would show "ZDR enforced" while the in-flight turn keeps
  // its already-captured routing — a per-turn promise the UI must not
  // appear to break. Plan detail loses to privacy correctness here.
  assert.match(
    getPrivacyToggleDisabledReason({ isStreaming: true }),
    /Wait for the current response to finish/,
  );
});

test('privacy toggle streaming guard takes priority over other busy reasons', () => {
  assert.match(
    getPrivacyToggleDisabledReason({ isStreaming: true, isUploading: true, isUpdatingPrivacy: true }),
    /Wait for the current response to finish/,
  );
});

test('privacy toggle blocks enabling ZDR off-provider but not disabling it', () => {
  assert.match(
    getPrivacyToggleDisabledReason({ isEnablingUnavailable: true }),
    /Requires OpenRouter/,
  );
  // Disable direction: caller passes isEnablingUnavailable=false when
  // effectiveZdr is already true, so the toggle stays clickable.
  assert.equal(getPrivacyToggleDisabledReason({ isEnablingUnavailable: false }), null);
});

test('resolveEffectiveZdr ignores the settings default off-provider but keeps explicit metadata', () => {
  // No conversation-level choice: a bare preference must not brick sends.
  assert.equal(
    resolveEffectiveZdr({ metadata: {} }, { zdrEnabled: true }, false),
    false,
  );
  assert.equal(
    resolveEffectiveZdr({ metadata: {} }, { zdrEnabled: true }, true),
    true,
  );
  // Explicit conversation metadata (the privacy promise) survives regardless
  // of provider availability — the backend rejects it loudly instead.
  assert.equal(
    resolveEffectiveZdr({ metadata: { zdr_enabled: true } }, {}, false),
    true,
  );
  assert.equal(
    resolveEffectiveZdr({ metadata: { zdr_enabled: false } }, { zdrEnabled: true }, false),
    false,
  );
});

test('attachment enhancement privacy prefers conversation mode over global setting', () => {
  assert.equal(resolveAttachmentEnhancementZdr(true, { zdrEnabled: false }), true);
  assert.equal(resolveAttachmentEnhancementZdr(false, { zdrEnabled: true }), false);
  assert.equal(resolveAttachmentEnhancementZdr(undefined, { zdrEnabled: true }), true);
});

test('formatCurrency keeps compact precision for small and large totals', () => {
  assert.equal(formatCurrency(null), '$0.00');
  assert.equal(formatCurrency(0.004), '$0.0040');
  assert.equal(formatCurrency(0.1249), '$0.12');
  assert.equal(formatCurrency(12), '$12.00');
});
