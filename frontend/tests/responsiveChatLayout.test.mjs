import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  getChatSurfaceClass,
  getTrustRowGridClass,
  getTrustRowCostTileClass,
  getStageTabListClass,
} from '../src/utils/responsiveChatLayout.js';

test('chat surfaces use the wider shared responsive content width', () => {
  assert.match(getChatSurfaceClass('messages'), /max-w-4xl/);
  assert.match(getChatSurfaceClass('composer'), /max-w-4xl/);
  assert.match(getChatSurfaceClass('composer'), /px-3/);
});

test('trust controls wrap within the available conversation column', () => {
  assert.match(getTrustRowGridClass(), /flex-wrap/);
  assert.doesNotMatch(getTrustRowGridClass(), /grid-cols/);
  assert.match(getTrustRowCostTileClass(), /h-8/);
});

test('stage tab lists scroll at narrow widths and wrap when space allows', () => {
  const tabClass = getStageTabListClass();
  assert.match(tabClass, /overflow-x-auto/);
  assert.match(tabClass, /sm:flex-wrap/);
});
