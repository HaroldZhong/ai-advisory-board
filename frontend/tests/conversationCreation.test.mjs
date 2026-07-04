import test from 'node:test';
import assert from 'node:assert/strict';

import { createConversationWithDefaults } from '../src/utils/conversationCreation.js';

test('creates a conversation with preset trust defaults atomically', async () => {
  const expectedConversation = { id: 'conv-1', title: 'New Conversation' };
  const calls = [];

  const result = await createConversationWithDefaults({
    apiClient: {
      async createConversation(topic, councilMembers, chairmanModel, options) {
        calls.push({ topic, councilMembers, chairmanModel, options });
        return expectedConversation;
      },
    },
    topic: 'New Conversation',
    councilMembers: null,
    chairmanModel: null,
    presetId: 'private',
    zdrEnabled: true,
    defaultSessionBudgetUsd: 2,
  });

  assert.equal(result, expectedConversation);
  assert.deepEqual(calls, [{
    topic: 'New Conversation',
    councilMembers: null,
    chairmanModel: null,
    options: {
      presetId: 'private',
      zdrEnabled: true,
      budgetUsd: 2,
      budgetAllowOverage: false,
      defaultMode: undefined,
    },
  }]);
});

test('passes default_mode through to the API client when provided', async () => {
  const expectedConversation = { id: 'conv-2', title: 'New Conversation' };
  const calls = [];

  await createConversationWithDefaults({
    apiClient: {
      async createConversation(topic, councilMembers, chairmanModel, options) {
        calls.push(options);
        return expectedConversation;
      },
    },
    topic: 'New Conversation',
    councilMembers: null,
    chairmanModel: 'anthropic/claude-opus-4.7',
    presetId: null,
    zdrEnabled: false,
    defaultSessionBudgetUsd: null,
    defaultMode: 'chat',
  });

  assert.equal(calls[0].defaultMode, 'chat');
});
