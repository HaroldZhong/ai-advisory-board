// frontend/tests/codeBlock.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { languageFromClassName } from '../src/utils/codeLanguage.js';
import { isSafeImageSrc } from '../src/utils/safeHref.js';

test('languageFromClassName extracts the language from a language-x class', () => {
  assert.equal(languageFromClassName('language-python'), 'python');
  assert.equal(languageFromClassName('language-js'), 'js');
});

test('languageFromClassName handles multiple classes in any order', () => {
  assert.equal(languageFromClassName('hljs language-python'), 'python');
  assert.equal(languageFromClassName('language-python hljs'), 'python');
});

test('languageFromClassName returns null when there is no language class', () => {
  assert.equal(languageFromClassName(undefined), null);
  assert.equal(languageFromClassName(''), null);
  assert.equal(languageFromClassName('hljs'), null);
  assert.equal(languageFromClassName('language-'), null);
});

test('isSafeImageSrc allows http(s) image URLs only', () => {
  assert.equal(isSafeImageSrc('https://example.com/cat.png'), true);
  assert.equal(isSafeImageSrc('http://example.com/cat.png'), true);
});

test('isSafeImageSrc blocks data/file/blob and other non-http(s) sources', () => {
  assert.equal(isSafeImageSrc('data:image/png;base64,iVBORw0KGgo='), false);
  assert.equal(isSafeImageSrc('file:///etc/passwd'), false);
  assert.equal(isSafeImageSrc('blob:https://example.com/uuid'), false);
  assert.equal(isSafeImageSrc('javascript:alert(1)'), false);
});

test('isSafeImageSrc blocks relative paths, anchors, and mailto (unlike isSafeHref)', () => {
  assert.equal(isSafeImageSrc('/local/cat.png'), false);
  assert.equal(isSafeImageSrc('#anchor'), false);
  assert.equal(isSafeImageSrc('mailto:a@b.com'), false);
  assert.equal(isSafeImageSrc(undefined), false);
});
