import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// No DOM/React renderer in this test runner, so the beforeunload guard
// (P3-T8 item 5) is verified at the source level: the listener must be
// wired to the isLoading effect dependency and use the standard
// preventDefault/returnValue idiom, not a homegrown one the desktop WebView
// or another browser might not honor.
function readHookSource() {
  const here = dirname(fileURLToPath(import.meta.url));
  return readFileSync(join(here, '../src/hooks/useStreamingConversation.js'), 'utf-8');
}

test('a beforeunload listener is registered and cleaned up around isLoading', () => {
  const source = readHookSource();
  assert.match(source, /addEventListener\('beforeunload'/);
  assert.match(source, /removeEventListener\('beforeunload'/);
  assert.match(source, /useEffect\(\(\) => \{\s*\n\s*if \(!isLoading\) return undefined;/);
});

test('the beforeunload handler uses the standard browser-native guard idiom', () => {
  const source = readHookSource();
  assert.match(source, /event\.preventDefault\(\)/);
  assert.match(source, /event\.returnValue = ''/);
});

test('stream errors reload the persisted conversation instead of keeping a phantom assistant', () => {
  const source = readHookSource();
  assert.match(source, /eventType === 'error'/);
  assert.match(source, /api\.getConversation\(targetConversationId\)/);
  assert.match(source, /prev\?\.id === targetConversationId \? persisted : prev/);
  assert.match(source, /\.finally\(\(\) => \{\s*setIsLoading\(false\);/);
});
