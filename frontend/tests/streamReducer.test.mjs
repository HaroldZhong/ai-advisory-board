import assert from 'node:assert/strict';
import test from 'node:test';

import { streamReducer } from '../src/utils/streamReducer.js';

const availableModels = [
  { id: 'model-a', pricing: { input: 1, output: 1 } },
  { id: 'model-b', pricing: { input: 1, output: 1 } },
];

function councilState(overrides = {}) {
  return {
    conversation: {
      id: 'conv-1',
      messages: [
        { role: 'user', content: 'Question' },
        {
          role: 'assistant',
          stage1: null,
          stage2: null,
          stage3: null,
          metadata: null,
          loading: { stage1: false, stage2: false, stage3: false, stage3_status: 'pending' },
          ...overrides.lastMessage,
        },
      ],
    },
    isLoading: true,
    budgetWarning: null,
  };
}

function chatState(overrides = {}) {
  return {
    conversation: {
      id: 'conv-1',
      messages: [
        { role: 'user', content: 'Question' },
        { role: 'assistant', content: '', loading: { chat: true }, ...overrides.lastMessage },
      ],
    },
    isLoading: true,
    budgetWarning: null,
  };
}

function lastMessage(state) {
  const messages = state.conversation.messages;
  return messages[messages.length - 1];
}

test('stage1_model_complete appends results incrementally, then stage1_complete replaces with full list', () => {
  const state = councilState();

  const afterFirst = streamReducer(
    state,
    { type: 'stage1_model_complete', data: { model: 'model-a', response: 'Answer A' }, index: 0 },
    { availableModels },
  );
  assert.deepEqual(lastMessage(afterFirst).stage1, [{ model: 'model-a', response: 'Answer A' }]);

  const afterSecond = streamReducer(
    afterFirst,
    { type: 'stage1_model_complete', data: { model: 'model-b', response: 'Answer B' }, index: 1 },
    { availableModels },
  );
  assert.deepEqual(lastMessage(afterSecond).stage1, [
    { model: 'model-a', response: 'Answer A' },
    { model: 'model-b', response: 'Answer B' },
  ]);

  const finalEvent = {
    type: 'stage1_complete',
    data: [
      { model: 'model-a', response: 'Answer A', usage: { prompt_tokens: 1000000, completion_tokens: 0 } },
      { model: 'model-b', response: 'Answer B', usage: { prompt_tokens: 1000000, completion_tokens: 0 } },
    ],
  };
  const afterComplete = streamReducer(afterSecond, finalEvent, { availableModels });
  assert.equal(lastMessage(afterComplete).stage1.length, 2);
  assert.equal(lastMessage(afterComplete).stage1[0].usage.prompt_tokens, 1000000);
  assert.equal(lastMessage(afterComplete).loading.stage1, false);
});

test('stage1_complete stores results, merges reasoning buffers, and accrues cost', () => {
  const state = councilState({
    lastMessage: { reasoningBuffers: { stage1: { 0: { text: 'Buffered reasoning' } } } },
  });
  const event = {
    type: 'stage1_complete',
    data: [{ model: 'model-a', response: 'Answer A', usage: { prompt_tokens: 1000000, completion_tokens: 0 } }],
  };

  const next = streamReducer(state, event, { availableModels });
  const msg = lastMessage(next);

  assert.equal(msg.stage1[0].reasoning, 'Buffered reasoning');
  assert.equal(msg.loading.stage1, false);
  assert.equal(msg.running_cost, 1);
});

test('stage2_complete stores results, metadata, and accrues cost', () => {
  const state = councilState({ lastMessage: { running_cost: 1 } });
  const event = {
    type: 'stage2_complete',
    data: [{ model: 'model-b', ranking: '1', usage: { prompt_tokens: 0, completion_tokens: 2000000 } }],
    metadata: { label_to_model: { A: 'model-a' }, aggregate_rankings: [] },
  };

  const next = streamReducer(state, event, { availableModels });
  const msg = lastMessage(next);

  assert.equal(msg.stage2[0].model, 'model-b');
  assert.deepEqual(msg.metadata, event.metadata);
  assert.equal(msg.loading.stage2, false);
  assert.equal(msg.running_cost, 3);
});

test('stage3_complete stores the synthesized result and accrues cost', () => {
  const state = councilState({ lastMessage: { running_cost: 3 } });
  const event = {
    type: 'stage3_complete',
    data: { model: 'model-a', response: 'Final answer', usage: { prompt_tokens: 1000000, completion_tokens: 0 } },
  };

  const next = streamReducer(state, event, { availableModels });
  const msg = lastMessage(next);

  assert.equal(msg.stage3.response, 'Final answer');
  assert.equal(msg.loading.stage3, false);
  assert.equal(msg.running_cost, 4);
});

test('chat_response sets content and optional reasoning, clears chat loading', () => {
  const state = chatState();
  const event = { type: 'chat_response', data: { content: 'Hello there', reasoning: 'Thinking...' } };

  const next = streamReducer(state, event, { availableModels });
  const msg = lastMessage(next);

  assert.equal(msg.content, 'Hello there');
  assert.equal(msg.reasoning, 'Thinking...');
  assert.equal(msg.loading.chat, false);
});

test('chat_response accepts a bare string payload', () => {
  const state = chatState();
  const event = { type: 'chat_response', data: 'Plain string content' };

  const next = streamReducer(state, event, { availableModels });
  assert.equal(lastMessage(next).content, 'Plain string content');
});

test('title_complete is a no-op on state (title reload is a hook side effect)', () => {
  const state = chatState();
  const event = { type: 'title_complete', data: { title: 'New Title' } };

  const next = streamReducer(state, event, { availableModels });
  assert.equal(next, state);
});

test('budget_warning sets budgetWarning from event data', () => {
  const state = chatState();
  const event = { type: 'budget_warning', data: { threshold: 0.8, percentage: 80 } };

  const next = streamReducer(state, event, { availableModels });
  assert.deepEqual(next.budgetWarning, { threshold: 0.8, percentage: 80 });
});

test('error marks the last assistant message interrupted and clears isLoading', () => {
  const state = councilState();
  const event = { type: 'error', message: 'boom' };

  const next = streamReducer(state, event, { availableModels });
  const msg = lastMessage(next);

  assert.equal(msg.loading.stage1, false);
  assert.equal(msg.loading.stage2, false);
  assert.equal(msg.loading.stage3, false);
  assert.equal(msg.loading.stage3_status, 'error');
  assert.equal(next.isLoading, false);
});

test('complete applies turn cost, conversation totals, and clears isLoading', () => {
  const state = councilState({ lastMessage: { running_cost: 4 } });
  const event = {
    type: 'complete',
    data: { turn_cost: 4, total_cost: 10, session_usage: { spent: 10 }, budget_spent_pct: 50 },
  };

  const next = streamReducer(state, event, { availableModels });

  assert.equal(lastMessage(next).running_cost, 4);
  assert.equal(next.conversation.total_cost, 10);
  assert.deepEqual(next.conversation.session_usage, { spent: 10 });
  assert.equal(next.conversation.budget_spent_pct, 50);
  assert.equal(next.isLoading, false);
});

test('edit_truncated is a no-op (backend-only bookkeeping event)', () => {
  const state = chatState();
  const event = { type: 'edit_truncated', data: { edit_index: 1, attachments: [] } };

  const next = streamReducer(state, event, { availableModels });
  assert.equal(next, state);
});

test('unknown event types are a no-op', () => {
  const state = chatState();
  const event = { type: 'some_future_event', data: { anything: true } };

  const next = streamReducer(state, event, { availableModels });
  assert.equal(next, state);
});
