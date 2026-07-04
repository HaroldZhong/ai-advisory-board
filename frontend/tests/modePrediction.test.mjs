import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { predictNextMessageMode } from '../src/utils/modePrediction.js';

test('empty conversation predicts council', () => {
  assert.equal(predictNextMessageMode({ messageCount: 0 }), 'council');
});

test('populated conversation predicts chat', () => {
  assert.equal(predictNextMessageMode({ messageCount: 4 }), 'chat');
});

test('edit back to message 0 predicts council even when messages exist', () => {
  assert.equal(predictNextMessageMode({ messageCount: 4, editIndex: 0 }), 'council');
});

test('edit mid-conversation predicts chat', () => {
  assert.equal(predictNextMessageMode({ messageCount: 4, editIndex: 2 }), 'chat');
});


test('stale editIndex beyond the stored count clamps to the real count', () => {
  assert.equal(predictNextMessageMode({ messageCount: 0, editIndex: 2 }), 'council');
});

test('no arguments defaults to council (new conversation)', () => {
  assert.equal(predictNextMessageMode(), 'council');
});

test('defaultMode "chat" always predicts chat, even on the first message', () => {
  assert.equal(predictNextMessageMode({ messageCount: 0, defaultMode: 'chat' }), 'chat');
});

test('defaultMode "chat" predicts chat on a follow-up too', () => {
  assert.equal(predictNextMessageMode({ messageCount: 4, defaultMode: 'chat' }), 'chat');
});

test('defaultMode "chat" overrides an edit back to message 0', () => {
  assert.equal(predictNextMessageMode({ messageCount: 4, editIndex: 0, defaultMode: 'chat' }), 'chat');
});

test('defaultMode "council" mirrors the legacy effective-count rule on first message', () => {
  assert.equal(predictNextMessageMode({ messageCount: 0, defaultMode: 'council' }), 'council');
});

test('defaultMode "council" mirrors the legacy effective-count rule on follow-up', () => {
  assert.equal(predictNextMessageMode({ messageCount: 4, defaultMode: 'council' }), 'chat');
});

test('the streaming hook sends mode auto and never a locally computed mode', () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const hookSource = readFileSync(
    join(here, '../src/hooks/useStreamingConversation.js'),
    'utf-8',
  );
  assert.match(hookSource, /'auto'/, 'hook must send the literal auto mode');
  assert.doesNotMatch(
    hookSource,
    /isFollowUp \? 'chat' : 'council'/,
    'hook must not compute the routing mode from message count',
  );
});
