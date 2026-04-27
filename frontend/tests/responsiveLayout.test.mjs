import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SIDEBAR_COLLAPSED_STORAGE_KEY,
  getSidebarMode,
  parseSidebarCollapsedPreference,
} from '../src/utils/responsiveLayout.js';

test('sidebar mode follows desktop breakpoints and user preference', () => {
  assert.equal(getSidebarMode(960, false), 'icon-only');
  assert.equal(getSidebarMode(1023, false), 'icon-only');
  assert.equal(getSidebarMode(1024, false), 'expanded');
  assert.equal(getSidebarMode(1280, false), 'expanded');

  assert.equal(getSidebarMode(1280, true), 'collapsed');
  assert.equal(getSidebarMode(960, true), 'icon-only');
});

test('sidebar preference parser only accepts explicit collapsed values', () => {
  assert.equal(SIDEBAR_COLLAPSED_STORAGE_KEY, 'aab.sidebar.collapsed');
  assert.equal(parseSidebarCollapsedPreference('true'), true);
  assert.equal(parseSidebarCollapsedPreference('false'), false);
  assert.equal(parseSidebarCollapsedPreference(null), false);
  assert.equal(parseSidebarCollapsedPreference('collapsed'), false);
});
