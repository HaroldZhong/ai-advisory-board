/**
 * Remember the last New Conversation dialog selection across sessions
 * (P3-T8 item 1) so reopening the dialog doesn't reset to the defaults
 * every time. Pure serialize/deserialize helpers, storage injectable for
 * tests — same shape as responsiveLayout.js's sidebar preference helpers.
 *
 * Deliberately excludes anything sensitive (API keys, etc.) and anything
 * that isn't really a "selection" (transient UI navigation like which tab
 * is active) — only the choices that should survive to the next open.
 */
export const MODEL_SELECTOR_STORAGE_KEY = 'aab.modelSelector.lastSelection';

const DEFAULTS = Object.freeze({
  conversationMode: 'chat',
  activeTab: 'presets',
  selectedPresetId: 'balanced',
  selectedCouncil: [],
  selectedChairman: '',
});

/**
 * Turn a persisted selection back into a well-formed object. Missing,
 * corrupt, or unknown-shaped input all fall back to defaults field-by-field
 * so a partially-valid record isn't discarded wholesale.
 */
export function deserializeModelSelectorSelection(raw) {
  if (typeof raw !== 'string' || !raw) return { ...DEFAULTS };

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ...DEFAULTS };
  }

  if (!parsed || typeof parsed !== 'object') return { ...DEFAULTS };

  return {
    conversationMode: parsed.conversationMode === 'council' ? 'council' : DEFAULTS.conversationMode,
    // Which tab (Presets vs Custom) was active when the selection was saved
    // (Codex review, P3-T8 round 3): without this, a saved custom council
    // reopened on the Presets tab and handleConfirm silently created a
    // preset conversation instead of honoring the saved custom ids.
    activeTab: parsed.activeTab === 'custom' ? 'custom' : DEFAULTS.activeTab,
    selectedPresetId: typeof parsed.selectedPresetId === 'string' && parsed.selectedPresetId
      ? parsed.selectedPresetId
      : DEFAULTS.selectedPresetId,
    selectedCouncil: Array.isArray(parsed.selectedCouncil)
      ? parsed.selectedCouncil.filter((id) => typeof id === 'string')
      : [...DEFAULTS.selectedCouncil],
    selectedChairman: typeof parsed.selectedChairman === 'string'
      ? parsed.selectedChairman
      : DEFAULTS.selectedChairman,
  };
}

/** Serialize only the known fields — unknown fields on the input are dropped, not round-tripped. */
export function serializeModelSelectorSelection(selection) {
  const { conversationMode, activeTab, selectedPresetId, selectedCouncil, selectedChairman } = {
    ...DEFAULTS,
    ...selection,
  };
  return JSON.stringify({ conversationMode, activeTab, selectedPresetId, selectedCouncil, selectedChairman });
}

export function readModelSelectorSelection(storage) {
  try {
    const targetStorage = storage ?? globalThis.localStorage;
    return deserializeModelSelectorSelection(targetStorage?.getItem(MODEL_SELECTOR_STORAGE_KEY));
  } catch {
    return { ...DEFAULTS };
  }
}

export function writeModelSelectorSelection(selection, storage) {
  try {
    const targetStorage = storage ?? globalThis.localStorage;
    targetStorage?.setItem(MODEL_SELECTOR_STORAGE_KEY, serializeModelSelectorSelection(selection));
  } catch {
    // Storage can be unavailable in privacy-restricted or sandboxed contexts.
  }
}
