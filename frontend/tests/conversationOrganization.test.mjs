import test from 'node:test';
import assert from 'node:assert/strict';

import { groupConversationsByFolder } from '../src/utils/conversationOrganization.js';

test('groups conversations by valid folder and keeps invalid folder ids in root', () => {
  const folders = [
    { id: 'folder-a', name: 'Research' },
    { id: 'folder-b', name: 'Planning' },
  ];
  const conversations = [
    { id: 'conv-1', folder_id: 'folder-a', title: 'A' },
    { id: 'conv-2', folder_id: 'deleted-folder', title: 'B' },
    { id: 'conv-3', folder_id: null, title: 'C' },
    { id: 'conv-4', folder_id: 'folder-b', title: 'D' },
  ];

  const grouped = groupConversationsByFolder(conversations, folders);

  assert.deepEqual(grouped.folderConversationMap, {
    'folder-a': [conversations[0]],
    'folder-b': [conversations[3]],
  });
  assert.deepEqual(grouped.unfolderedConversations, [
    conversations[1],
    conversations[2],
  ]);
});
