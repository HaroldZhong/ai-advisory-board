import test from 'node:test';
import assert from 'node:assert/strict';

import {
  THINKING_EFFORT_LEVELS,
  formatThinkingEffortLabel,
  getThinkingEffortOption,
  getThinkingEffortTone,
  isValidThinkingEffort,
  mergeConversationThinkingEffortUpdate,
  resolveEffectiveThinkingEffort,
  setConversationThinkingEffortMetadata,
} from '../src/utils/thinkingEffort.js';

test('thinking effort levels match backend contract order', () => {
  assert.deepEqual(THINKING_EFFORT_LEVELS, ['minimal', 'low', 'medium', 'high', 'xhigh']);
});

test('thinking effort display labels are user-readable', () => {
  assert.equal(formatThinkingEffortLabel('minimal'), 'Minimal');
  assert.equal(formatThinkingEffortLabel('medium'), 'Medium');
  assert.equal(formatThinkingEffortLabel('xhigh'), 'X-High');
  assert.equal(formatThinkingEffortLabel('unknown'), 'Medium');
});

test('thinking effort resolver prefers conversation metadata over preset defaults', () => {
  assert.equal(
    resolveEffectiveThinkingEffort({
      metadata: { thinking_effort: 'high', default_reasoning_effort: 'low' },
    }),
    'high',
  );

  assert.equal(
    resolveEffectiveThinkingEffort({
      metadata: { preset_id: 'research', default_reasoning_effort: 'high' },
    }),
    'high',
  );

  assert.equal(
    resolveEffectiveThinkingEffort({
      metadata: { preset_id: 'custom', thinking_effort: 'turbo' },
    }),
    'medium',
  );

  assert.equal(resolveEffectiveThinkingEffort(null), 'medium');
});

test('v1.3.0 B3: the budget preset no longer caps thinking effort', () => {
  // A Budget-preset conversation that stored High keeps High -- the preset is a
  // suggestion, never a cap (the silent clamp to medium is gone).
  assert.equal(resolveEffectiveThinkingEffort({
    metadata: { preset_id: 'budget', thinking_effort: 'high' },
  }), 'high');
});

test('thinking effort options include honest relative cost hints', () => {
  assert.equal(getThinkingEffortOption('minimal').costHint, '~0.2x reasoning tokens');
  assert.equal(getThinkingEffortOption('medium').costHint, 'Default reasoning token budget');
  assert.equal(getThinkingEffortOption('xhigh').costHint, '~1.9x reasoning tokens');
});

test('thinking effort tones escalate only for expensive settings', () => {
  assert.equal(getThinkingEffortTone('minimal'), 'neutral');
  assert.equal(getThinkingEffortTone('medium'), 'neutral');
  assert.equal(getThinkingEffortTone('high'), 'warn');
  assert.equal(getThinkingEffortTone('xhigh'), 'danger');
});

test('thinking effort validation is strict', () => {
  assert.equal(isValidThinkingEffort('low'), true);
  assert.equal(isValidThinkingEffort('xhigh'), true);
  assert.equal(isValidThinkingEffort(''), false);
  assert.equal(isValidThinkingEffort('turbo'), false);
  assert.equal(isValidThinkingEffort(null), false);
});

test('local thinking effort metadata update only applies to the active conversation', () => {
  const current = {
    id: 'conversation-1',
    messages: [],
    metadata: { preset_id: 'balanced', thinking_effort: 'medium' },
  };

  const updated = setConversationThinkingEffortMetadata(current, 'conversation-1', 'high');
  assert.notEqual(updated, current);
  assert.equal(updated.metadata.thinking_effort, 'high');
  assert.equal(updated.metadata.preset_id, 'balanced');

  const budgetUpdated = setConversationThinkingEffortMetadata(
    {
      id: 'conversation-1',
      messages: [],
      metadata: { preset_id: 'budget', thinking_effort: 'low' },
    },
    'conversation-1',
    'xhigh',
  );
  // v1.3.0 B3: no preset cap -- X-High is stored as X-High for a Budget conversation.
  assert.equal(budgetUpdated.metadata.thinking_effort, 'xhigh');

  assert.equal(
    setConversationThinkingEffortMetadata(current, 'conversation-2', 'high'),
    current,
  );

  const unchanged = setConversationThinkingEffortMetadata(updated, 'conversation-1', 'turbo');
  assert.equal(unchanged, updated);

  const reset = setConversationThinkingEffortMetadata(updated, 'conversation-1', undefined);
  assert.notEqual(reset, updated);
  assert.equal(reset.metadata.thinking_effort, undefined);
  assert.equal(reset.metadata.preset_id, 'balanced');
});

test('thinking effort update merge preserves local streaming messages', () => {
  const current = {
    id: 'conversation-1',
    messages: [{ role: 'assistant', content: '', loading: { chat: true } }],
    metadata: { preset_id: 'balanced', thinking_effort: 'medium' },
  };
  const serverSnapshot = {
    id: 'conversation-1',
    messages: [],
    metadata: { preset_id: 'balanced', thinking_effort: 'xhigh' },
  };

  const merged = mergeConversationThinkingEffortUpdate(current, serverSnapshot);

  assert.equal(merged.messages, current.messages);
  assert.equal(merged.metadata.thinking_effort, 'xhigh');
});

test('thinking effort update merge ignores stale conversations', () => {
  const current = {
    id: 'conversation-2',
    messages: [],
    metadata: { preset_id: 'balanced', thinking_effort: 'low' },
  };
  const staleServerSnapshot = {
    id: 'conversation-1',
    metadata: { thinking_effort: 'xhigh' },
  };

  assert.equal(mergeConversationThinkingEffortUpdate(current, staleServerSnapshot), current);
  assert.equal(mergeConversationThinkingEffortUpdate(null, staleServerSnapshot), null);
});
