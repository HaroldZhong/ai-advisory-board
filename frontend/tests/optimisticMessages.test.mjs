import test from 'node:test';
import assert from 'node:assert/strict';

import {
  rollbackFailedSendConversation,
  rollbackFailedSendMessages,
  reconcileInterruptedRun,
} from '../src/utils/optimisticMessages.js';

test('stop keeps unconfirmed content and ignores unrelated or late persisted replies', () => {
  const scope = { conversationId: 'a', requestId: 'request-a', runId: 'run-a', reason: 'Stop requested' };
  const partial = { role: 'assistant', content: '可复制的部分正文', client_request_id: 'request-a', loading: { chat: true } };
  const conversation = { id: 'a', messages: [partial] };
  const stopped = reconcileInterruptedRun(conversation, { id: 'a', messages: [{ role: 'assistant', content: 'Other', metadata: { run_id: 'run-b' } }] }, scope);
  assert.equal(stopped.messages[0].content, partial.content);
  assert.equal(stopped.messages[0].unconfirmed, true);
  assert.equal(stopped.messages[0].loading.chat, false);
  assert.equal(reconcileInterruptedRun({ ...conversation, id: 'b' }, null, scope).id, 'b');
  const newer = { id: 'a', messages: [{ ...partial, client_request_id: 'new-request' }] };
  assert.equal(reconcileInterruptedRun(newer, null, scope), newer);
  const saved = { role: 'assistant', content: 'Saved', metadata: { run_id: 'run-a', persistence_state: 'saved' } };
  const reconciled = reconcileInterruptedRun(stopped, { id: 'a', messages: [saved] }, scope);
  assert.equal(reconciled.messages[0].content, 'Saved');
  assert.equal(reconciled.messages[0].unconfirmed, undefined);
});

test('failed edit rollback restores the original message tail', () => {
  const originalMessages = [
    { role: 'user', content: 'Question 1' },
    { role: 'assistant', content: 'Answer 1' },
    { role: 'user', content: 'Question 2' },
    { role: 'assistant', content: 'Answer 2' },
  ];
  const truncatedWithOptimisticPair = [
    originalMessages[0],
    { role: 'user', content: 'Edited question' },
    { role: 'assistant', loading: { chat: true } },
  ];

  assert.deepEqual(
    rollbackFailedSendMessages(truncatedWithOptimisticPair, {
      editIndex: 1,
      previousMessages: originalMessages,
    }),
    originalMessages,
  );
});

test('failed non-edit rollback removes only the optimistic user and assistant pair', () => {
  const messages = [
    { role: 'user', content: 'Question 1' },
    { role: 'assistant', content: 'Answer 1' },
    { role: 'user', content: 'New question' },
    { role: 'assistant', loading: { chat: true } },
  ];

  assert.deepEqual(
    rollbackFailedSendMessages(messages, { editIndex: -1, previousMessages: null }),
    messages.slice(0, 2),
  );
});

test('failed send rollback does not mutate a newly active conversation', () => {
  const activeConversation = {
    id: 'conversation-b',
    messages: [
      { role: 'user', content: 'Different thread question' },
      { role: 'assistant', content: 'Different thread answer' },
    ],
  };
  const originalConversationA = [
    { role: 'user', content: 'Original question' },
    { role: 'assistant', content: 'Original answer' },
  ];

  assert.equal(
    rollbackFailedSendConversation(activeConversation, {
      conversationId: 'conversation-a',
      editIndex: 0,
      previousMessages: originalConversationA,
    }),
    activeConversation,
  );
});


test('an interrupted unsaved prompt restores persisted history without registering draft attachments', () => {
  const scope = { conversationId: 'a', requestId: 'request-a', runId: 'run-a', reason: 'Stop requested' };
  const user = { role: 'user', content: 'Unsaved edit', client_request_id: 'request-a', attachments: [{ attachment_id: 'new' }] };
  const assistant = { role: 'assistant', content: 'Draft', client_request_id: 'request-a', loading: { chat: true } };
  const optimistic = { id: 'a', messages: [user, assistant] };
  const persisted = { id: 'a', messages: [{ role: 'user', content: 'Original', attachment_ids: ['old'] }] };
  assert.deepEqual(reconcileInterruptedRun(optimistic, persisted, scope).messages, persisted.messages);
  assert.deepEqual(reconcileInterruptedRun(optimistic, { id: 'a', messages: [] }, scope).messages, []);
  // The separately retained answer still preserves its content when no prompt was saved.
  const draft = reconcileInterruptedRun({ id: 'a', messages: [assistant] }, persisted, scope).messages[0];
  assert.equal(draft.content, 'Draft');
  assert.equal(draft.unconfirmed, true);
  // A failed reconciliation read cannot establish what was persisted.
  assert.equal(reconcileInterruptedRun(optimistic, null, scope).messages[0], user);
  const newer = { id: 'a', messages: [...optimistic.messages, { role: 'user', client_request_id: 'newer' }] };
  assert.equal(reconcileInterruptedRun(newer, persisted, scope), newer);
});

test('interruption keeps the user message actually saved for this run', () => {
  const scope = { conversationId: 'a', requestId: 'request-a', runId: 'run-a', reason: 'Connection lost' };
  const user = { role: 'user', content: 'Question', client_request_id: 'request-a', attachments: [{ attachment_id: 'new' }] };
  const assistant = { role: 'assistant', content: 'Draft', client_request_id: 'request-a' };
  const savedUser = { role: 'user', content: 'Question', attachment_ids: ['new'], metadata: { run_id: 'run-a' } };
  const result = reconcileInterruptedRun({ id: 'a', messages: [user, assistant] }, { id: 'a', messages: [savedUser] }, scope);
  assert.deepEqual(result.messages[0], { ...savedUser, client_request_id: 'request-a' });
  assert.equal(result.messages[1].unconfirmed, true);
});
