export async function createConversationWithDefaults({
  apiClient,
  topic,
  councilMembers,
  chairmanModel,
  presetId,
  zdrEnabled,
  defaultSessionBudgetUsd,
}) {
  return apiClient.createConversation(topic, councilMembers, chairmanModel, {
    presetId,
    zdrEnabled,
    budgetUsd: defaultSessionBudgetUsd,
    budgetAllowOverage: false,
  });
}
