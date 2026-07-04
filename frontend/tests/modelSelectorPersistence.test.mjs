import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MODEL_SELECTOR_STORAGE_KEY,
  deserializeModelSelectorSelection,
  serializeModelSelectorSelection,
  readModelSelectorSelection,
  writeModelSelectorSelection,
} from '../src/utils/modelSelectorPersistence.js';

function createMapStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    values,
  };
}

const DEFAULT_SELECTION = {
  conversationMode: 'chat',
  activeTab: 'presets',
  selectedPresetId: 'balanced',
  selectedCouncil: [],
  selectedChairman: '',
};

test('round-trips a full selection through serialize/deserialize', () => {
  const selection = {
    conversationMode: 'council',
    activeTab: 'presets',
    selectedPresetId: 'private',
    selectedCouncil: ['a/model-1', 'b/model-2'],
    selectedChairman: 'a/model-1',
  };
  const restored = deserializeModelSelectorSelection(serializeModelSelectorSelection(selection));
  assert.deepEqual(restored, selection);
});

test('round-trips a saved Custom-tab selection (Codex review, round 3)', () => {
  const selection = {
    conversationMode: 'council',
    activeTab: 'custom',
    selectedPresetId: 'balanced',
    selectedCouncil: ['a/model-1', 'b/model-2', 'c/model-3'],
    selectedChairman: 'a/model-1',
  };
  const restored = deserializeModelSelectorSelection(serializeModelSelectorSelection(selection));
  assert.deepEqual(restored, selection);
  assert.equal(restored.activeTab, 'custom');
});

test('corrupt JSON falls back to defaults', () => {
  const restored = deserializeModelSelectorSelection('{not-json');
  assert.deepEqual(restored, DEFAULT_SELECTION);
});

test('unknown fields are ignored, known fields still applied', () => {
  const restored = deserializeModelSelectorSelection(JSON.stringify({
    conversationMode: 'council',
    selectedPresetId: 'research',
    apiKey: 'sk-should-be-dropped',
    somethingElse: 42,
  }));
  assert.equal(restored.conversationMode, 'council');
  assert.equal(restored.selectedPresetId, 'research');
  assert.equal('apiKey' in restored, false);
  assert.equal('somethingElse' in restored, false);
});

test('missing localStorage entry falls back to defaults', () => {
  assert.deepEqual(deserializeModelSelectorSelection(null), DEFAULT_SELECTION);
  assert.deepEqual(deserializeModelSelectorSelection(undefined), DEFAULT_SELECTION);
});

test('an invalid conversationMode value falls back to the chat default', () => {
  const restored = deserializeModelSelectorSelection(JSON.stringify({ conversationMode: 'bogus' }));
  assert.equal(restored.conversationMode, 'chat');
});

test('an invalid activeTab value falls back to the presets default', () => {
  const restored = deserializeModelSelectorSelection(JSON.stringify({ activeTab: 'bogus' }));
  assert.equal(restored.activeTab, 'presets');
});

test('a missing activeTab value falls back to the presets default', () => {
  const restored = deserializeModelSelectorSelection(JSON.stringify({ conversationMode: 'council' }));
  assert.equal(restored.activeTab, 'presets');
});

test('a non-array selectedCouncil falls back to an empty array', () => {
  const restored = deserializeModelSelectorSelection(JSON.stringify({ selectedCouncil: 'not-an-array' }));
  assert.deepEqual(restored.selectedCouncil, []);
});

test('read/write round-trip through an injected storage', () => {
  const storage = createMapStorage();
  const selection = {
    conversationMode: 'council',
    activeTab: 'custom',
    selectedPresetId: 'budget',
    selectedCouncil: ['x/1'],
    selectedChairman: 'x/1',
  };

  writeModelSelectorSelection(selection, storage);
  assert.equal(typeof storage.values.get(MODEL_SELECTOR_STORAGE_KEY), 'string');
  assert.deepEqual(readModelSelectorSelection(storage), selection);
});

test('reading with no stored value returns defaults', () => {
  const storage = createMapStorage();
  assert.deepEqual(readModelSelectorSelection(storage), DEFAULT_SELECTION);
});

test('read/write tolerate a throwing storage', () => {
  const throwingStorage = {
    getItem() { throw new Error('blocked'); },
    setItem() { throw new Error('blocked'); },
  };
  assert.deepEqual(readModelSelectorSelection(throwingStorage), DEFAULT_SELECTION);
  assert.doesNotThrow(() => writeModelSelectorSelection({ conversationMode: 'council' }, throwingStorage));
});

test('serialize never includes an apiKey or other unknown field', () => {
  const json = serializeModelSelectorSelection({
    conversationMode: 'chat',
    selectedPresetId: 'balanced',
    selectedCouncil: [],
    selectedChairman: '',
    apiKey: 'sk-super-secret',
  });
  assert.equal(json.includes('apiKey'), false);
  assert.equal(json.includes('sk-super-secret'), false);
});
