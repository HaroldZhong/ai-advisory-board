import test from 'node:test';
import assert from 'node:assert/strict';

import {
  FIRST_RUN_BUDGET_PRESETS,
  looksLikeOpenRouterKey,
  buildFirstRunSettings,
} from '../src/utils/firstRunSetup.js';

test('validates OpenRouter key shape without accepting empty strings', () => {
  assert.equal(looksLikeOpenRouterKey(''), false);
  assert.equal(looksLikeOpenRouterKey('sk-test'), false);
  assert.equal(looksLikeOpenRouterKey(' sk-or-v1-abcdefghijklmnop '), true);
});

test('builds persisted first-run settings from privacy and budget choices', () => {
  assert.deepEqual(buildFirstRunSettings({ zdrChoice: 'on', budgetUsd: 2 }), {
    defaultZdrEnabled: true,
    zdrEnabled: true,
    defaultSessionBudgetUsd: 2,
  });
});

test('budget presets expose the recommended two-dollar default', () => {
  const recommended = FIRST_RUN_BUDGET_PRESETS.find((preset) => preset.recommended);
  assert.equal(recommended.value, 2);
});
