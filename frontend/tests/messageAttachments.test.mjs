import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { applyStreamUpdateToActiveConversation } from '../src/utils/reasoningMessages.js';
import { rollbackFailedSendConversation } from '../src/utils/optimisticMessages.js';
import { resolveSendMode } from '../src/utils/modePrediction.js';
import { normalizeAdvancedSettingsForMode } from '../src/utils/advancedSettingsAvailability.js';
import { resolveEffectiveZdr } from '../src/utils/trustState.js';
import { extractMessageAttachmentIds } from '../src/utils/messageAttachments.js';

test('prefers the raw attachment_ids list when present', () => {
  const message = {
    role: 'user',
    content: 'See attached',
    attachment_ids: ['att-1', 'att-2'],
    attachments: [{ attachment_id: 'att-1' }, { attachment_id: 'att-2' }],
  };
  assert.deepEqual(extractMessageAttachmentIds(message), ['att-1', 'att-2']);
});

test('derives ids from the attachments metadata list when attachment_ids is absent', () => {
  const message = {
    role: 'user',
    content: 'See attached',
    attachments: [{ attachment_id: 'att-1', filename: 'a.pdf' }, { attachment_id: 'att-2', filename: 'b.pdf' }],
  };
  assert.deepEqual(extractMessageAttachmentIds(message), ['att-1', 'att-2']);
});

test('drops metadata entries with no attachment_id', () => {
  const message = { attachments: [{ attachment_id: 'att-1' }, { filename: 'no-id.pdf' }, null] };
  assert.deepEqual(extractMessageAttachmentIds(message), ['att-1']);
});

test('returns an empty array for a message with no attachments', () => {
  assert.deepEqual(extractMessageAttachmentIds({ role: 'user', content: 'plain text' }), []);
});

test('returns an empty array for null/undefined input', () => {
  assert.deepEqual(extractMessageAttachmentIds(null), []);
  assert.deepEqual(extractMessageAttachmentIds(undefined), []);
});

// Execute the component's actual edit handler without adding a React/DOM runner.
const editHandler = readFileSync(new URL('../src/components/ChatInterface.jsx', import.meta.url), 'utf8')
  .match(/const handleEditSubmit = async \(\) => \{[\s\S]*?\n  \};/)[0];

for (const [label, selection] of [['inherit', null], ['none', []], ['subset', ['att-2']]]) {
  test(`regeneration sends the ${label} material scope with the original attachments`, async () => {
    const state = { index: 2, content: 'Revised question', ids: ['att-1', 'att-2'], metadata: [] };
    const calls = [];
    await runInNewContext(`${editHandler}\nhandleEditSubmit()`, {
      conversation: { id: 'a' },
      conversationIdRef: { current: 'a' },
      setSendError: () => {},
      editingIndex: state.index,
      editingContent: state.content,
      editingAttachmentIds: state.ids,
      editingAttachmentMetadata: state.metadata,
      evidenceSourceIds: selection,
      budgetCapBlock: { blocked: false },
      setEditingIndex: value => { state.index = value; },
      setEditingContent: value => { state.content = value; },
      setEditingAttachmentIds: value => { state.ids = value; },
      setEditingAttachmentMetadata: value => { state.metadata = value; },
      onSendMessage: async (...args) => { calls.push(args); },
    });
    assert.equal(calls.length, 1);
    assert.equal(calls[0][0], 'Revised question');
    assert.deepEqual(calls[0][1], ['att-1', 'att-2']);
    assert.equal(calls[0][3], 2);
    assert.deepEqual(calls[0][4]?.evidenceSourceIds, selection);
    assert.equal(state.index, -1);
  });
}

const sendHandler = readFileSync(new URL('../src/hooks/useStreamingConversation.js', import.meta.url), 'utf8')
  .match(/const sendMessage = async [\s\S]*?\n  \};/)[0];

for (const status of [400, 403, 409, 412]) {
  test(`rejected regeneration (${status}) rolls back history and preserves the edit draft`, async () => {
    const original = [{ role: 'user', content: 'Original' }, { role: 'assistant', content: 'Saved answer' }];
    let conversation = { id: 'a', messages: original };
    const error = Object.assign(new Error('Selection rejected'), { status });
    const sendMessage = runInNewContext(`${sendHandler}\nsendMessage`, {
      conversationId: 'a', currentConversation: conversation,
      activeRequest: { current: null }, latestRequest: { current: null },
      crypto: { randomUUID: () => 'test-request' }, AbortController,
      setCurrentConversation: update => { conversation = update(conversation); },
      applyStreamUpdateToActiveConversation, rollbackFailedSendConversation,
      resolveSendMode, normalizeAdvancedSettingsForMode, resolveEffectiveZdr,
      setIsLoading: () => {}, setStreamStatus: () => {}, settings: {}, zdrAvailable: true,
      api: { sendMessageStream: async () => { throw error; } },
      toast: () => { throw new Error('Preflight errors belong to the composer'); },
    });
    const draft = { index: 0, content: 'Revised draft', ids: ['att-1'], metadata: [{ attachment_id: 'att-1' }] };
    const expected = structuredClone(draft);
    let displayedError;
    let budgetOpened = false;
    await runInNewContext(`${editHandler}\nhandleEditSubmit()`, {
      conversation, conversationIdRef: { current: 'a' },
      editingIndex: draft.index, editingContent: draft.content,
      editingAttachmentIds: draft.ids, editingAttachmentMetadata: draft.metadata,
      evidenceSourceIds: ['later-source'], budgetCapBlock: { blocked: false },
      setEditingIndex: value => { draft.index = value; },
      setEditingContent: value => { draft.content = value; },
      setEditingAttachmentIds: value => { draft.ids = value; },
      setEditingAttachmentMetadata: value => { draft.metadata = value; },
      setSendError: value => { displayedError = value; },
      setShowBudgetSelector: value => { budgetOpened = value; },
      onSendMessage: sendMessage,
    });
    assert.deepEqual(conversation.messages, original);
    assert.deepEqual(draft, expected);
    assert.equal(displayedError, error.message);
    assert.equal(budgetOpened, status === 409);
  });
}
