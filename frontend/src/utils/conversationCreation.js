export async function createConversationWithDefaults({
  apiClient,
  topic,
  councilMembers,
  chairmanModel,
  defaultSessionBudgetUsd,
  onBudgetError,
}) {
  const conversation = await apiClient.createConversation(topic, councilMembers, chairmanModel);

  if (defaultSessionBudgetUsd != null) {
    try {
      await apiClient.updateSessionPolicy(conversation.id, {
        budget_usd: defaultSessionBudgetUsd,
      });
    } catch (error) {
      onBudgetError?.(error, conversation);
    }
  }

  return conversation;
}
