import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildConfigStatusFailureState,
  buildConfigStatusSuccessState,
  getConfigStatusRetryDelayMs,
  isZdrAvailableForProvider,
} from '../src/utils/configStatus.js';

test('shows first-run setup when status check succeeds with no API key', () => {
  assert.deepEqual(buildConfigStatusSuccessState({ has_api_key: false, provider_kind: 'openrouter' }), {
    configStatus: { loading: false, hasApiKey: false, providerKind: 'openrouter', error: null },
    showFirstRunSetup: true,
  });
});

test('does not show first-run setup when status check transport fails', () => {
  assert.deepEqual(buildConfigStatusFailureState(), {
    configStatus: { loading: false, hasApiKey: null, providerKind: 'openrouter', error: 'unavailable' },
    showFirstRunSetup: false,
  });
});

test('does not show first-run setup when API key exists', () => {
  assert.deepEqual(buildConfigStatusSuccessState({ has_api_key: true, provider_kind: 'openrouter' }), {
    configStatus: { loading: false, hasApiKey: true, providerKind: 'openrouter', error: null },
    showFirstRunSetup: false,
  });
});

test('defaults provider kind to openrouter when the field is missing', () => {
  const result = buildConfigStatusSuccessState({ has_api_key: true });
  assert.equal(result.configStatus.providerKind, 'openrouter');
});

test('marks provider kind from status response', () => {
  const result = buildConfigStatusSuccessState({ has_api_key: true, provider_kind: 'openai-compatible' });
  assert.equal(result.configStatus.providerKind, 'openai-compatible');
});

test('ZDR is only available on openrouter', () => {
  assert.equal(isZdrAvailableForProvider('openrouter'), true);
  assert.equal(isZdrAvailableForProvider('openai-compatible'), false);
  assert.equal(isZdrAvailableForProvider(undefined), false);
});

test('backs off config status retries with a cap', () => {
  assert.equal(getConfigStatusRetryDelayMs(0), 1000);
  assert.equal(getConfigStatusRetryDelayMs(1), 2000);
  assert.equal(getConfigStatusRetryDelayMs(2), 4000);
  assert.equal(getConfigStatusRetryDelayMs(3), 5000);
  assert.equal(getConfigStatusRetryDelayMs(10), 5000);
});
