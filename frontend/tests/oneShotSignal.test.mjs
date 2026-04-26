import test from 'node:test';
import assert from 'node:assert/strict';

import { shouldConsumeOneShotSignal } from '../src/utils/oneShotSignal.js';

test('opens once for a new positive signal', () => {
  assert.equal(shouldConsumeOneShotSignal(1, 0), true);
});

test('does not reopen for an already consumed signal after remount', () => {
  assert.equal(shouldConsumeOneShotSignal(1, 1), false);
});

test('ignores the initial zero signal', () => {
  assert.equal(shouldConsumeOneShotSignal(0, 0), false);
});
