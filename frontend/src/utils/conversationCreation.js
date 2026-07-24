export async function createConversationWithDefaults({
  apiClient,
  topic,
  councilMembers,
  chairmanModel,
  presetId,
  zdrEnabled,
  defaultSessionBudgetUsd,
  defaultMode,
}) {
  return apiClient.createConversation(topic, councilMembers, chairmanModel, {
    presetId,
    zdrEnabled,
    budgetUsd: defaultSessionBudgetUsd,
    // v1.3.0 D3: new conversations allow overage by default (warn, don't block).
    budgetAllowOverage: true,
    defaultMode,
  });
}
