import test from 'node:test';
import assert from 'node:assert/strict';

import {
  rollbackFailedSendConversation,
  rollbackFailedSendMessages,
} from '../src/utils/optimisticMessages.js';

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
