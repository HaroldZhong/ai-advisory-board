import test from 'node:test';
import assert from 'node:assert/strict';

import {
  FIRST_RUN_BUDGET_PRESETS,
  looksLikeOpenRouterKey,
  isAcceptableApiKey,
  buildFirstRunSettings,
  mapConnectivityResult,
} from '../src/utils/firstRunSetup.js';

test('validates OpenRouter key shape without accepting empty strings', () => {
  assert.equal(looksLikeOpenRouterKey(''), false);
  assert.equal(looksLikeOpenRouterKey('sk-test'), false);
  assert.equal(looksLikeOpenRouterKey(' sk-or-v1-abcdefghijklmnop '), true);
});

test('key acceptance is provider-aware', () => {
  // openrouter: existing sk-or- shape check applies
  assert.equal(isAcceptableApiKey('sk-or-v1-abcdefghijklmnop', 'openrouter'), true);
  assert.equal(isAcceptableApiKey('junk', 'openrouter'), false);
  assert.equal(isAcceptableApiKey('', 'openrouter'), false);

  // openai-compatible: any non-empty trimmed value is accepted
  assert.equal(isAcceptableApiKey('anything', 'openai-compatible'), true);
  assert.equal(isAcceptableApiKey('ollama-local', 'openai-compatible'), true);
  assert.equal(isAcceptableApiKey('  ', 'openai-compatible'), false);
  assert.equal(isAcceptableApiKey('', 'openai-compatible'), false);
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

test('maps a reachable network with a valid key to connected', () => {
  assert.deepEqual(
    mapConnectivityResult({ reachable: true, key_valid: true, error_kind: null, detail: '' }),
    { status: 'connected', message: 'Connected to OpenRouter.' }
  );
});

test('maps a reachable network with an invalid key to bad_key using backend detail', () => {
  assert.deepEqual(
    mapConnectivityResult({
      reachable: true,
      key_valid: false,
      error_kind: 'invalid_key',
      detail: 'OpenRouter rejected this API key.',
    }),
    { status: 'bad_key', message: 'OpenRouter rejected this API key.' }
  );
});

test('maps an unreachable network to blocked using backend detail with proxy hint', () => {
  assert.deepEqual(
    mapConnectivityResult({
      reachable: false,
      key_valid: null,
      error_kind: 'network_blocked',
      detail: 'Could not reach OpenRouter. If you are behind a restrictive network, configure a proxy.',
    }),
    {
      status: 'blocked',
      message: 'Could not reach OpenRouter. If you are behind a restrictive network, configure a proxy.',
    }
  );
});

test('maps a reachable network with an unchecked key to key_unchecked', () => {
  assert.deepEqual(
    mapConnectivityResult({ reachable: true, key_valid: null, error_kind: null, detail: '' }),
    { status: 'key_unchecked', message: 'Network OK. API key not checked yet.' }
  );
});

test('maps a missing/failed fetch body to blocked with a generic message', () => {
  assert.deepEqual(mapConnectivityResult(null), {
    status: 'blocked',
    message: 'Could not reach the backend to test the connection.',
  });
  assert.deepEqual(mapConnectivityResult(undefined), {
    status: 'blocked',
    message: 'Could not reach the backend to test the connection.',
  });
});
