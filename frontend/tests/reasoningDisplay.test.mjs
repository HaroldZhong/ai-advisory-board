import assert from 'node:assert/strict';
import test from 'node:test';

import {
  formatDuration,
  formatTokenCount,
  getReasoningStatusLabel,
  hasReasoningText,
} from '../src/utils/reasoningDisplay.js';

test('formatTokenCount keeps small counts readable', () => {
  assert.equal(formatTokenCount(0), '0 tokens');
  assert.equal(formatTokenCount(999), '999 tokens');
  assert.equal(formatTokenCount(null), null);
});

test('formatTokenCount abbreviates thousands and millions', () => {
  assert.equal(formatTokenCount(1200), '1.2k tokens');
  assert.equal(formatTokenCount(1250000), '1.3M tokens');
});

test('formatDuration uses compact human-readable units', () => {
  assert.equal(formatDuration(850), '850ms');
  assert.equal(formatDuration(1500), '1.5s');
  assert.equal(formatDuration(65000), '1m 5s');
  assert.equal(formatDuration(null), null);
});

test('formatDuration normalizes second rollover at minute boundaries', () => {
  assert.equal(formatDuration(119999), '2m 0s');
});

test('getReasoningStatusLabel maps stream states to user-facing copy', () => {
  assert.equal(getReasoningStatusLabel('streaming'), 'Reasoning');
  assert.equal(getReasoningStatusLabel('complete'), 'Reasoning complete');
  assert.equal(getReasoningStatusLabel('unavailable'), 'Reasoning unavailable');
  assert.equal(getReasoningStatusLabel('unknown'), 'Reasoning complete');
});

test('hasReasoningText rejects blank reasoning', () => {
  assert.equal(hasReasoningText('  \n '), false);
  assert.equal(hasReasoningText('Model considered the trade-offs.'), true);
});
