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

test('trust row wraps before switching to dense desktop columns', () => {
  const gridClass = getTrustRowGridClass();
  assert.match(gridClass, /grid-cols-1/);
  assert.match(gridClass, /sm:grid-cols-2/);
  assert.match(gridClass, /lg:grid-cols-\[1fr_1fr_1\.25fr_1fr_auto\]/);

  const costClass = getTrustRowCostTileClass();
  assert.match(costClass, /sm:col-span-2/);
  assert.match(costClass, /lg:col-span-1/);
});

test('stage tab lists scroll at narrow widths and wrap when space allows', () => {
  const tabClass = getStageTabListClass();
  assert.match(tabClass, /overflow-x-auto/);
  assert.match(tabClass, /sm:flex-wrap/);
});
