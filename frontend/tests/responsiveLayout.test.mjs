import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SIDEBAR_COLLAPSED_STORAGE_KEY,
  getSidebarMode,
  parseSidebarCollapsedPreference,
  readSidebarCollapsedPreference,
  writeSidebarCollapsedPreference,
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

test('sidebar preference storage helpers tolerate unavailable storage', () => {
  const throwingStorage = {
    getItem() {
      throw new Error('storage blocked');
    },
    setItem() {
      throw new Error('storage blocked');
    },
  };

  assert.equal(readSidebarCollapsedPreference(throwingStorage), false);
  assert.doesNotThrow(() => writeSidebarCollapsedPreference(true, throwingStorage));
});

test('sidebar preference storage helpers tolerate blocked global storage access', () => {
  const originalDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get() {
      throw new Error('global storage blocked');
    },
  });

  try {
    assert.equal(readSidebarCollapsedPreference(), false);
    assert.doesNotThrow(() => writeSidebarCollapsedPreference(true));
  } finally {
    if (originalDescriptor) {
      Object.defineProperty(globalThis, 'localStorage', originalDescriptor);
    } else {
      delete globalThis.localStorage;
    }
  }
});

test('sidebar preference storage helpers read and write explicit values', () => {
  const values = new Map();
  const storage = {
    getItem(key) {
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      values.set(key, value);
    },
  };

  writeSidebarCollapsedPreference(true, storage);
  assert.equal(values.get(SIDEBAR_COLLAPSED_STORAGE_KEY), 'true');
  assert.equal(readSidebarCollapsedPreference(storage), true);

  writeSidebarCollapsedPreference(false, storage);
  assert.equal(values.get(SIDEBAR_COLLAPSED_STORAGE_KEY), 'false');
  assert.equal(readSidebarCollapsedPreference(storage), false);
});
