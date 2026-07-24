import assert from 'node:assert/strict';
import test from 'node:test';

import {
  formatDuration,
  formatReasoningActuals,
  formatTokenCount,
  getReasoningStatusLabel,
  hasReasoningText,
} from '../src/utils/reasoningDisplay.js';

test('B5/E3 §3d: reasoning actuals are keyed on token count, never on text', () => {
  // real reasoning tokens -> the count
  assert.equal(formatReasoningActuals(1200), 'reasoning: 1.2k tokens');
  assert.equal(formatReasoningActuals(900), 'reasoning: 900 tokens');
  // the honesty trap: 0 tokens (even if reasoning text was returned) -> not available
  assert.equal(formatReasoningActuals(0), 'reasoning: not available');
  // absent / null / undefined / NaN -> not available (never implies "no reasoning")
  assert.equal(formatReasoningActuals(null), 'reasoning: not available');
  assert.equal(formatReasoningActuals(undefined), 'reasoning: not available');
  assert.equal(formatReasoningActuals(NaN), 'reasoning: not available');
});

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
