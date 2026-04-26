import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildConfigStatusFailureState,
  buildConfigStatusSuccessState,
  getConfigStatusRetryDelayMs,
} from '../src/utils/configStatus.js';

test('shows first-run setup when status check succeeds with no API key', () => {
  assert.deepEqual(buildConfigStatusSuccessState({ has_api_key: false }), {
    configStatus: { loading: false, hasApiKey: false, error: null },
    showFirstRunSetup: true,
  });
});

test('does not show first-run setup when status check transport fails', () => {
  assert.deepEqual(buildConfigStatusFailureState(), {
    configStatus: { loading: false, hasApiKey: null, error: 'unavailable' },
    showFirstRunSetup: false,
  });
});

test('does not show first-run setup when API key exists', () => {
  assert.deepEqual(buildConfigStatusSuccessState({ has_api_key: true }), {
    configStatus: { loading: false, hasApiKey: true, error: null },
    showFirstRunSetup: false,
  });
});

test('backs off config status retries with a cap', () => {
  assert.equal(getConfigStatusRetryDelayMs(0), 1000);
  assert.equal(getConfigStatusRetryDelayMs(1), 2000);
  assert.equal(getConfigStatusRetryDelayMs(2), 4000);
  assert.equal(getConfigStatusRetryDelayMs(3), 5000);
  assert.equal(getConfigStatusRetryDelayMs(10), 5000);
});
