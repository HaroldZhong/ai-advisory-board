export function groupConversationsByFolder(conversations, folders) {
  const folderIds = new Set(folders.map((folder) => folder.id));
  const folderConversationMap = {};
  const unfolderedConversations = [];

  for (const conversation of conversations) {
    const folderId = conversation.folder_id;
    if (folderId && folderIds.has(folderId)) {
      if (!folderConversationMap[folderId]) {
        folderConversationMap[folderId] = [];
      }
      folderConversationMap[folderId].push(conversation);
    } else {
      unfolderedConversations.push(conversation);
    }
  }

  return {
    folderConversationMap,
    unfolderedConversations,
  };
}
