export function getAdvancedSettingAvailability(nextMessageMode) {
  const isCouncilMode = nextMessageMode === 'council';

  return {
    executionMode: { disabled: isCouncilMode },
    ragPreset: { disabled: isCouncilMode },
    modelTier: { disabled: false },
    notice: isCouncilMode
      ? 'Execution mode and context level only apply to chat mode; the next message is council mode and will run the full council pipeline.'
      : null,
  };
}

export function normalizeAdvancedSettingsForMode(settings, nextMessageMode) {
  if (nextMessageMode !== 'council') {
    return settings;
  }

  return {
    ...settings,
    executionMode: 'auto',
    ragPreset: 'auto',
  };
}

export function isChatOnlyAdvancedOptionDisabled(field, optionId, nextMessageMode) {
  if (nextMessageMode !== 'council') {
    return false;
  }
  if (!['executionMode', 'ragPreset'].includes(field)) {
    return false;
  }
  return optionId !== 'auto';
}
