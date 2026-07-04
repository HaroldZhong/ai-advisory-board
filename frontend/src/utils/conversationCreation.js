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
    budgetAllowOverage: false,
    defaultMode,
  });
}
