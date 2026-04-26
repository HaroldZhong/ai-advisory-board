export function buildConfigStatusSuccessState(status) {
  const hasApiKey = Boolean(status?.has_api_key);

  return {
    configStatus: { loading: false, hasApiKey, error: null },
    showFirstRunSetup: !hasApiKey,
  };
}

export function buildConfigStatusFailureState() {
  return {
    configStatus: { loading: false, hasApiKey: null, error: 'unavailable' },
    showFirstRunSetup: false,
  };
}

export function getConfigStatusRetryDelayMs(attempt) {
  return Math.min(1000 * (2 ** attempt), 5000);
}
