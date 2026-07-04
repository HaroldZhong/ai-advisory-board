import test from 'node:test';
import assert from 'node:assert/strict';
import { extractMessageAttachmentIds } from '../src/utils/messageAttachments.js';

test('prefers the raw attachment_ids list when present', () => {
  const message = {
    role: 'user',
    content: 'See attached',
    attachment_ids: ['att-1', 'att-2'],
    attachments: [{ attachment_id: 'att-1' }, { attachment_id: 'att-2' }],
  };
  assert.deepEqual(extractMessageAttachmentIds(message), ['att-1', 'att-2']);
});

test('derives ids from the attachments metadata list when attachment_ids is absent', () => {
  const message = {
    role: 'user',
    content: 'See attached',
    attachments: [{ attachment_id: 'att-1', filename: 'a.pdf' }, { attachment_id: 'att-2', filename: 'b.pdf' }],
  };
  assert.deepEqual(extractMessageAttachmentIds(message), ['att-1', 'att-2']);
});

test('drops metadata entries with no attachment_id', () => {
  const message = { attachments: [{ attachment_id: 'att-1' }, { filename: 'no-id.pdf' }, null] };
  assert.deepEqual(extractMessageAttachmentIds(message), ['att-1']);
});

test('returns an empty array for a message with no attachments', () => {
  assert.deepEqual(extractMessageAttachmentIds({ role: 'user', content: 'plain text' }), []);
});

test('returns an empty array for null/undefined input', () => {
  assert.deepEqual(extractMessageAttachmentIds(null), []);
  assert.deepEqual(extractMessageAttachmentIds(undefined), []);
});
