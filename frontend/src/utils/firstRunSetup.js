export const FIRST_RUN_BUDGET_PRESETS = [
  { id: 'none', label: 'No limit', value: null, description: 'No session budget' },
  { id: 'light', label: '$1', value: 1, description: 'Light exploration' },
  { id: 'standard', label: '$2', value: 2, description: 'Balanced starting point', recommended: true },
  { id: 'research', label: '$5', value: 5, description: 'Extended research' },
];

export function looksLikeOpenRouterKey(value) {
  return typeof value === 'string' && /^sk-or-[A-Za-z0-9_-]{12,}/.test(value.trim());
}

export function buildFirstRunSettings({ zdrChoice, budgetUsd }) {
  return {
    defaultZdrEnabled: zdrChoice === 'on',
    zdrEnabled: zdrChoice === 'on',
    defaultSessionBudgetUsd: budgetUsd ?? null,
  };
}

/**
 * Map a /api/config/connectivity response body to UI status + message.
 * Handles a missing body (e.g. the fetch itself failed) as 'blocked'.
 */
export function mapConnectivityResult(body) {
  if (!body) {
    return { status: 'blocked', message: 'Could not reach the backend to test the connection.' };
  }

  const { reachable, key_valid: keyValid, detail } = body;

  if (!reachable) {
    return { status: 'blocked', message: detail };
  }
  if (keyValid === true) {
    return { status: 'connected', message: 'Connected to OpenRouter.' };
  }
  if (keyValid === false) {
    return { status: 'bad_key', message: detail };
  }
  return { status: 'key_unchecked', message: 'Network OK. API key not checked yet.' };
}
