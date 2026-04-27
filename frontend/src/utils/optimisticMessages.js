export function rollbackFailedSendMessages(
  messages,
  { editIndex = -1, previousMessages = null } = {},
) {
  if (editIndex >= 0 && Array.isArray(previousMessages)) {
    return [...previousMessages];
  }

  const nextMessages = [...(messages || [])];
  if (nextMessages.length >= 2 && nextMessages[nextMessages.length - 1]?.role === 'assistant') {
    nextMessages.splice(-2);
  } else if (nextMessages.length >= 1 && nextMessages[nextMessages.length - 1]?.role === 'user') {
    nextMessages.splice(-1);
  }
  return nextMessages;
}

export function rollbackFailedSendConversation(
  conversation,
  { conversationId, editIndex = -1, previousMessages = null } = {},
) {
  if (!conversation || conversation.id !== conversationId) {
    return conversation;
  }

  return {
    ...conversation,
    messages: rollbackFailedSendMessages(conversation.messages, {
      editIndex,
      previousMessages,
    }),
  };
}
