import test from 'node:test';
import assert from 'node:assert/strict';

import {
  estimateSelectionCost,
  filterModelsForRole,
  canConfirmModelSelection,
  canStartPresetWithZdr,
  getEffectivePresetZdr,
  isPresetAvailableForZdr,
  resolvePresetModels,
  resolveInitialZdrPreference,
} from '../src/utils/modelPresets.js';

const models = [
  {
    id: 'chair-zdr',
    name: 'Provider: Chair ZDR',
    type: 'chairman',
    supports_zdr: true,
    pricing: { input: 2, output: 10 },
  },
  {
    id: 'chair-open',
    name: 'Provider: Chair Open',
    type: 'chairman',
    supports_zdr: false,
    pricing: { input: 1, output: 3 },
  },
  {
    id: 'council-zdr',
    name: 'Provider: Council ZDR',
    type: 'council',
    supports_zdr: true,
    pricing: { input: 1, output: 4 },
  },
  {
    id: 'council-open',
    name: 'Provider: Council Open',
    type: 'council',
    supports_zdr: false,
    pricing: { input: 0.5, output: 1 },
  },
];

test('ZDR availability rejects presets with non-ZDR model IDs', () => {
  assert.equal(isPresetAvailableForZdr({
    chairman_model: 'chair-zdr',
    council_models: ['council-zdr'],
  }, models), true);

  assert.equal(isPresetAvailableForZdr({
    chairman_model: 'chair-open',
    council_models: ['council-zdr'],
  }, models), false);
});

test('required-ZDR presets force effective ZDR before submit', () => {
  const privatePreset = {
    requires_zdr: true,
    chairman_model: 'chair-zdr',
    council_models: ['council-zdr'],
  };

  assert.equal(getEffectivePresetZdr(privatePreset, false), true);
  assert.equal(canStartPresetWithZdr(privatePreset, models, false), true);
});

test('initial ZDR preference preserves legacy ZDR-enabled settings', () => {
  assert.equal(resolveInitialZdrPreference({
    defaultZdrEnabled: false,
    zdrEnabled: true,
  }), true);

  assert.equal(resolveInitialZdrPreference({
    defaultZdrEnabled: true,
    zdrEnabled: false,
  }), true);

  assert.equal(resolveInitialZdrPreference({
    defaultZdrEnabled: false,
    zdrEnabled: false,
  }), false);
});

test('ZDR-enabled presets cannot start with incompatible models', () => {
  const mixedPreset = {
    requires_zdr: false,
    chairman_model: 'chair-zdr',
    council_models: ['council-zdr', 'council-open'],
  };

  assert.equal(getEffectivePresetZdr(mixedPreset, true), true);
  assert.equal(canStartPresetWithZdr(mixedPreset, models, true), false);
});

test('model selection cannot confirm while loading or errored', () => {
  const baseSelection = {
    chairman: models[0],
    council: [models[2], models[2], models[2]],
    selectedPresetAvailable: true,
    minCouncilSize: 3,
  };

  assert.equal(canConfirmModelSelection({
    ...baseSelection,
    loading: false,
    error: null,
  }), true);

  assert.equal(canConfirmModelSelection({
    ...baseSelection,
    loading: true,
    error: null,
  }), false);

  assert.equal(canConfirmModelSelection({
    ...baseSelection,
    loading: false,
    error: 'Failed to load models',
  }), false);
});

test('resolvePresetModels counts hidden models when ZDR is enabled', () => {
  const result = resolvePresetModels({
    chairman_model: 'chair-zdr',
    council_models: ['council-zdr', 'council-open'],
  }, models, true);

  assert.equal(result.chairman.id, 'chair-zdr');
  assert.deepEqual(result.council.map((model) => model.id), ['council-zdr']);
  assert.equal(result.hiddenByZdr, 1);
});

test('estimateSelectionCost uses chairman and council pricing', () => {
  const cost = estimateSelectionCost({
    chairman: models[0],
    council: [models[2]],
  });

  assert.equal(cost, 0.0185);
});

test('filterModelsForRole applies role and ZDR filters', () => {
  assert.deepEqual(
    filterModelsForRole(models, 'chairman', true).map((model) => model.id),
    ['chair-zdr'],
  );
  assert.deepEqual(
    filterModelsForRole(models, 'council', false).map((model) => model.id),
    ['council-zdr', 'council-open'],
  );
});
