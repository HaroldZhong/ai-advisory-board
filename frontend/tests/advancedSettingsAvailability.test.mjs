import test from 'node:test';
import assert from 'node:assert/strict';

import {
  getAdvancedSettingAvailability,
  normalizeAdvancedSettingsForMode,
} from '../src/utils/advancedSettingsAvailability.js';

test('council mode marks chat-only advanced settings as disabled', () => {
  const availability = getAdvancedSettingAvailability('council');

  assert.equal(availability.executionMode.disabled, true);
  assert.equal(availability.ragPreset.disabled, true);
  assert.equal(availability.modelTier.disabled, false);
  assert.match(availability.notice, /council mode/i);
});

test('chat mode keeps all advanced settings editable', () => {
  const availability = getAdvancedSettingAvailability('chat');

  assert.equal(availability.executionMode.disabled, false);
  assert.equal(availability.ragPreset.disabled, false);
  assert.equal(availability.modelTier.disabled, false);
  assert.equal(availability.notice, null);
});

test('council mode normalizes chat-only overrides before save', () => {
  const normalized = normalizeAdvancedSettingsForMode(
    {
      executionMode: 'research',
      ragPreset: 'max',
      modelTier: 'premium',
      zdrEnabled: true,
      customInstructions: 'Use direct prose.',
    },
    'council',
  );

  assert.equal(normalized.executionMode, 'auto');
  assert.equal(normalized.ragPreset, 'auto');
  assert.equal(normalized.modelTier, 'premium');
  assert.equal(normalized.zdrEnabled, true);
  assert.equal(normalized.customInstructions, 'Use direct prose.');
});
