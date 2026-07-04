// frontend/tests/streamErrors.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { formatStreamErrorMessage } from '../src/utils/streamErrors.js';

test('empty/undefined message gets a generic fallback', () => {
  assert.match(formatStreamErrorMessage(undefined), /something went wrong/i);
  assert.match(formatStreamErrorMessage(''), /something went wrong/i);
});

test('network-flavored backend errors get an actionable hint', () => {
  const msg = formatStreamErrorMessage('ConnectError: [Errno 8] nodename nor servname provided');
  assert.match(msg, /network|connect/i);
  assert.match(msg, /openrouter/i);
});

test('other messages pass through, truncated to 300 chars', () => {
  const long = 'x'.repeat(500);
  const msg = formatStreamErrorMessage(long);
  assert.ok(msg.length <= 300);
});
