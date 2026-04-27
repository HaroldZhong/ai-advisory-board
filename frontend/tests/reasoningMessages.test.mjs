import assert from 'node:assert/strict';
import test from 'node:test';

import {
  appendContentDeltaToMessage,
  appendReasoningDeltaToMessage,
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
