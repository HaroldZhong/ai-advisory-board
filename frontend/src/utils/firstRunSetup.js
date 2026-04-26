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
