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
