import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ADVANCED_ROUTING_DISCLOSURE_STORAGE_KEY,
  getAdvancedRoutingSummary,
  getModelTierHint,
  getRagPresetHint,
  readAdvancedRoutingDisclosurePreference,
  writeAdvancedRoutingDisclosurePreference,
} from '../src/utils/advancedRoutingControls.js';

function createMemoryStorage(seed = {}) {
  const store = new Map(Object.entries(seed));
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
    removeItem(key) {
      store.delete(key);
    },
  };
}

test('advanced routing disclosure preference uses a namespaced key', () => {
  assert.equal(ADVANCED_ROUTING_DISCLOSURE_STORAGE_KEY, 'aab.advancedRouting.expanded');
});

test('advanced routing disclosure preference defaults closed and persists explicit choices', () => {
  const storage = createMemoryStorage();

  assert.equal(readAdvancedRoutingDisclosurePreference(storage), false);

  writeAdvancedRoutingDisclosurePreference(true, storage);
  assert.equal(readAdvancedRoutingDisclosurePreference(storage), true);

  writeAdvancedRoutingDisclosurePreference(false, storage);
  assert.equal(readAdvancedRoutingDisclosurePreference(storage), false);
});

test('advanced routing disclosure preference tolerates unavailable storage', () => {
  const blockedStorage = {
    getItem() {
      throw new Error('blocked');
    },
    setItem() {
      throw new Error('blocked');
    },
  };

  assert.equal(readAdvancedRoutingDisclosurePreference(blockedStorage), false);
  assert.doesNotThrow(() => writeAdvancedRoutingDisclosurePreference(true, blockedStorage));
});

test('advanced routing summaries expose hidden non-auto values', () => {
  assert.equal(
    getAdvancedRoutingSummary({ ragPreset: 'auto', modelTier: 'auto' }),
    'Context: Auto · Model tier: Auto',
  );
  assert.equal(
    getAdvancedRoutingSummary({ ragPreset: 'max', modelTier: 'premium' }),
    'Context: Maximum · Model tier: Premium',
  );
});

test('routing hints describe cost impact without fake precision', () => {
  assert.match(getRagPresetHint('auto'), /budget-aware/i);
  assert.match(getRagPresetHint('high'), /higher cost/i);
  assert.match(getRagPresetHint('max'), /32k context/i);
  assert.match(getRagPresetHint('max'), /not always better/i);
  assert.match(getModelTierHint('auto'), /selected preset/i);
  assert.match(getModelTierHint('premium'), /highest cost/i);
});
