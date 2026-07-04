// frontend/tests/safeHref.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { isSafeHref } from '../src/utils/safeHref.js';

test('allows normal web and relative links', () => {
  assert.equal(isSafeHref('https://example.com/x'), true);
  assert.equal(isSafeHref('http://example.com'), true);
  assert.equal(isSafeHref('mailto:a@b.com'), true);
  assert.equal(isSafeHref('/local/path'), true);
  assert.equal(isSafeHref('#anchor'), true);
});

test('blocks script-scheme and data URLs, case/whitespace tricks included', () => {
  assert.equal(isSafeHref('javascript:alert(1)'), false);
  assert.equal(isSafeHref('JaVaScRiPt:alert(1)'), false);
  assert.equal(isSafeHref('  javascript:alert(1)'), false);
  assert.equal(isSafeHref('java\tscript:alert(1)'), false);
  assert.equal(isSafeHref('java\nscript:alert(1)'), false);
  assert.equal(isSafeHref('data:text/html,<script>1</script>'), false);
  assert.equal(isSafeHref('vbscript:msgbox'), false);
  assert.equal(isSafeHref(undefined), false);
});

test('allowlist rejects every non-allowlisted scheme, not just known-bad ones', () => {
  assert.equal(isSafeHref('file:///etc/passwd'), false);
  assert.equal(isSafeHref('blob:https://example.com/uuid'), false);
  assert.equal(isSafeHref('intent://scan/#Intent;end'), false);
  assert.equal(isSafeHref('ms-msdt:/id'), false);
  assert.equal(isSafeHref('ftp://example.com/x'), false);
});
