import test from 'node:test';
import assert from 'node:assert/strict';

import { createConversationWithDefaults } from '../src/utils/conversationCreation.js';

test('returns the created conversation when applying the default budget fails', async () => {
  const expectedConversation = { id: 'conv-1', title: 'New Conversation' };
  const budgetErrors = [];

  const result = await createConversationWithDefaults({
    apiClient: {
      async createConversation() {
        return expectedConversation;
      },
      async updateSessionPolicy() {
        throw new Error('budget policy unavailable');
      },
    },
    topic: 'New Conversation',
    councilMembers: ['model-a'],
    chairmanModel: 'model-b',
    defaultSessionBudgetUsd: 2,
    onBudgetError: (error, conversation) => {
      budgetErrors.push({ error, conversation });
    },
  });

  assert.equal(result, expectedConversation);
  assert.equal(budgetErrors.length, 1);
  assert.equal(budgetErrors[0].conversation, expectedConversation);
  assert.equal(budgetErrors[0].error.message, 'budget policy unavailable');
});

test('applies the default session budget after conversation creation', async () => {
  const policyUpdates = [];

  const result = await createConversationWithDefaults({
    apiClient: {
      async createConversation(topic, councilMembers, chairmanModel) {
        return { id: 'conv-2', topic, councilMembers, chairmanModel };
      },
      async updateSessionPolicy(conversationId, policy) {
        policyUpdates.push({ conversationId, policy });
      },
    },
    topic: 'New Conversation',
    councilMembers: ['model-a'],
    chairmanModel: 'model-b',
    defaultSessionBudgetUsd: 5,
  });

  assert.equal(result.id, 'conv-2');
  assert.deepEqual(policyUpdates, [
    { conversationId: 'conv-2', policy: { budget_usd: 5 } },
  ]);
});
