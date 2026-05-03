export function shouldBlockConversationNavigation({
  currentConversationId,
  targetConversationId,
  isLoading,
}) {
  return Boolean(isLoading && currentConversationId && targetConversationId !== currentConversationId);
}

export function shouldBlockNewConversation(isLoading) {
  return Boolean(isLoading);
}

export function getBlockedConversationNavigationMessage() {
  return 'Wait for the current response to finish before switching conversations.';
}
