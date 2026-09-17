import test from 'node:test';
import assert from 'node:assert/strict';
import { shouldSubmitOnEnter } from '../src/utils/composerKeys.js';

test('Enter submits only outside IME composition and Shift+Enter remains a newline', () => {
  assert.equal(shouldSubmitOnEnter({ key: 'Enter' }), true);
  for (const event of [
    { key: 'Enter', shiftKey: true },
    { key: 'Enter', nativeEvent: { isComposing: true } },
    { key: 'Enter', isComposing: true },
    { key: 'Enter', nativeEvent: { isComposing: false, keyCode: 229 } },
    { key: 'Enter', keyCode: 229 },
    { key: 'Escape' },
  ]) assert.equal(shouldSubmitOnEnter(event), false, JSON.stringify(event));
});
