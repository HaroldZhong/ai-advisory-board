import test from 'node:test';
import assert from 'node:assert/strict';

import { isLandingOnly, resolveAppEntryTarget } from '../src/utils/appMode.js';

test('isLandingOnly is true when VITE_LANDING_ONLY is "true"', () => {
  assert.equal(isLandingOnly({ VITE_LANDING_ONLY: 'true' }), true);
});

test('isLandingOnly is false when VITE_LANDING_ONLY is "false"', () => {
  assert.equal(isLandingOnly({ VITE_LANDING_ONLY: 'false' }), false);
});

test('isLandingOnly is false when VITE_LANDING_ONLY is undefined', () => {
  assert.equal(isLandingOnly({}), false);
});

test('resolveAppEntryTarget points to the GitHub releases page in landing-only mode', () => {
  assert.equal(
    resolveAppEntryTarget({ VITE_LANDING_ONLY: 'true' }),
    'https://github.com/HaroldZhong/ai-advisory-board/releases'
  );
});

test('resolveAppEntryTarget points to /app when landing-only mode is off', () => {
  assert.equal(resolveAppEntryTarget({ VITE_LANDING_ONLY: 'false' }), '/app');
});

test('resolveAppEntryTarget defaults to /app when the flag is unset', () => {
  assert.equal(resolveAppEntryTarget({}), '/app');
});
