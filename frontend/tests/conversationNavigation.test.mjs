import { strict as assert } from 'node:assert';
import test from 'node:test';

import {
  getBlockedConversationNavigationMessage,
  shouldBlockConversationNavigation,
  shouldBlockNewConversation,
} from '../src/utils/conversationNavigation.js';

test('shouldBlockConversationNavigation blocks switching away during an active response', () => {
  assert.equal(shouldBlockConversationNavigation({
    currentConversationId: 'conv-a',
    targetConversationId: 'conv-b',
    isLoading: true,
  }), true);
});

test('shouldBlockConversationNavigation allows reselecting the active conversation while loading', () => {
  assert.equal(shouldBlockConversationNavigation({
    currentConversationId: 'conv-a',
    targetConversationId: 'conv-a',
    isLoading: true,
  }), false);
});

test('shouldBlockConversationNavigation allows switching when idle', () => {
  assert.equal(shouldBlockConversationNavigation({
    currentConversationId: 'conv-a',
    targetConversationId: 'conv-b',
    isLoading: false,
  }), false);
});

test('getBlockedConversationNavigationMessage explains why navigation is blocked', () => {
  assert.match(getBlockedConversationNavigationMessage(), /response.*finish/i);
});

test('shouldBlockNewConversation follows active response state', () => {
  assert.equal(shouldBlockNewConversation(true), true);
  assert.equal(shouldBlockNewConversation(false), false);
});
