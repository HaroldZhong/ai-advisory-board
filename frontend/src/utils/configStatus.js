export function buildConfigStatusSuccessState(status) {
  const hasApiKey = Boolean(status?.has_api_key);
  const providerKind = status?.provider_kind || 'openrouter';

  return {
    configStatus: { loading: false, hasApiKey, providerKind, error: null },
    showFirstRunSetup: !hasApiKey,
  };
}

export function buildConfigStatusFailureState() {
  return {
    configStatus: { loading: false, hasApiKey: null, providerKind: 'openrouter', error: 'unavailable' },
    showFirstRunSetup: false,
  };
}

export function isZdrAvailableForProvider(providerKind) {
  return providerKind === 'openrouter';
}

export function getConfigStatusRetryDelayMs(attempt) {
  return Math.min(1000 * (2 ** attempt), 5000);
}
