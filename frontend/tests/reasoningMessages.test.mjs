import assert from 'node:assert/strict';
import test from 'node:test';

import {
  appendContentDeltaToMessage,
  appendReasoningDeltaToMessage,
  applyStreamUpdateToActiveConversation,
  markLastAssistantStreamInterrupted,
  mergeReasoningBufferIntoResult,
  mergeReasoningBuffersIntoResults,
} from '../src/utils/reasoningMessages.js';

test('appendReasoningDeltaToMessage appends chat reasoning to the assistant message', () => {
  const message = { role: 'assistant', reasoning: 'First ' };

  const updated = appendReasoningDeltaToMessage(message, {
    scope: 'chat',
    stage: 'chat',
    text: 'second',
  });

  assert.equal(updated.reasoning, 'First second');
  assert.equal(message.reasoning, 'First ');
});

test('appendReasoningDeltaToMessage buffers council reasoning by stage and index', () => {
  const message = { role: 'assistant' };

  const updated = appendReasoningDeltaToMessage(message, {
    scope: 'council',
    stage: 'stage1',
    index: 1,
    model: 'provider/model-b',
    text: 'Reasoning',
  });

  assert.deepEqual(updated.reasoningBuffers.stage1['1'], {
    index: 1,
    model: 'provider/model-b',
    text: 'Reasoning',
  });
});

test('appendReasoningDeltaToMessage keeps index zero as an explicit slot', () => {
  const message = { role: 'assistant' };

  const updated = appendReasoningDeltaToMessage(message, {
    scope: 'council',
    stage: 'stage1',
    index: 0,
    model: 'provider/model-a',
    text: 'First reasoning',
  });

  assert.equal(updated.reasoningBuffers.stage1['0'].text, 'First reasoning');
});

test('appendReasoningDeltaToMessage warns and leaves message unchanged for unknown council stages', () => {
  const originalWarn = console.warn;
  const warnings = [];
  console.warn = (...args) => warnings.push(args.join(' '));

  try {
    const message = { role: 'assistant' };
    const updated = appendReasoningDeltaToMessage(message, {
      scope: 'council',
      stage: 'stageX',
      text: 'Unexpected',
    });

    assert.equal(updated, message);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /Unknown reasoning stream stage/);
  } finally {
    console.warn = originalWarn;
  }
});

test('mergeReasoningBuffersIntoResults fills missing reasoning without overwriting completed reasoning', () => {
  const results = [
    { model: 'model-a', response: 'A' },
    { model: 'model-b', response: 'B', reasoning: 'Completed B' },
  ];
  const buffers = {
    0: { text: 'Buffered A' },
    1: { text: 'Buffered B' },
  };

  const merged = mergeReasoningBuffersIntoResults(results, buffers);

  assert.equal(merged[0].reasoning, 'Buffered A');
  assert.equal(merged[1].reasoning, 'Completed B');
  assert.equal(results[0].reasoning, undefined);
});

test('mergeReasoningBufferIntoResult falls back to the first stage3 buffer', () => {
  const merged = mergeReasoningBufferIntoResult(
    { model: 'chairman', response: 'Final' },
    { default: { text: 'Chairman reasoning' } },
  );

  assert.equal(merged.reasoning, 'Chairman reasoning');
});

test('appendContentDeltaToMessage appends chat content only', () => {
  const message = { role: 'assistant', content: 'Hello' };

  assert.equal(
    appendContentDeltaToMessage(message, { stage: 'chat', text: ' world' }).content,
    'Hello world',
  );
  assert.equal(
    appendContentDeltaToMessage(message, { stage: 'stage1', text: 'ignored' }).content,
    'Hello',
  );
});

test('applyStreamUpdateToActiveConversation ignores stale events for inactive conversations', () => {
  const activeConversation = {
    id: 'conversation-b',
    messages: [{ role: 'assistant', content: 'Existing' }],
  };

  const updated = applyStreamUpdateToActiveConversation(
    activeConversation,
    'conversation-a',
    (conversation) => ({
      ...conversation,
      messages: [{ role: 'assistant', content: 'Corrupted' }],
    }),
  );

  assert.equal(updated, activeConversation);
});

test('applyStreamUpdateToActiveConversation applies updates for the originating conversation', () => {
  const activeConversation = {
    id: 'conversation-a',
    messages: [{ role: 'assistant', content: 'Existing' }],
  };

  const updated = applyStreamUpdateToActiveConversation(
    activeConversation,
    'conversation-a',
    (conversation) => ({
      ...conversation,
      messages: [{ role: 'assistant', content: 'Updated' }],
    }),
  );

  assert.equal(updated.messages[0].content, 'Updated');
});

test('markLastAssistantStreamInterrupted clears active assistant loading flags', () => {
  const conversation = {
    id: 'conversation-a',
    messages: [
      { role: 'user', content: 'Question' },
      {
        role: 'assistant',
        loading: {
          chat: true,
          stage1: true,
          stage2: false,
          stage3: true,
          stage3_status: 'pending',
        },
      },
    ],
  };

  const updated = markLastAssistantStreamInterrupted(conversation);

  assert.deepEqual(updated.messages[1].loading, {
    chat: false,
    stage1: false,
    stage2: false,
    stage3: false,
    stage3_status: 'error',
  });
  assert.equal(conversation.messages[1].loading.chat, true);
});
