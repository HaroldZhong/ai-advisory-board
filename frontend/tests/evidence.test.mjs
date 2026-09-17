import test from 'node:test';
import assert from 'node:assert/strict';
import { remarkEvidenceLinks, getTurnPresentation } from '../src/utils/evidence.js';

test('citations link only against their message snapshot, excluding code and existing links', () => {
  const tree = { type: 'root', children: [
    { type: 'paragraph', children: [{ type: 'text', value: 'Known [S1.1], unknown [S9.9]' }] },
    { type: 'code', value: '[S1.1]' },
    { type: 'link', url: 'https://example.com', children: [{ type: 'text', value: '[S1.1]' }] },
  ] };
  remarkEvidenceLinks({ citations: { '[S1.1]': {} } })()(tree);
  assert.equal(tree.children[0].children[1].url, '#aab-citation-S1.1');
  assert.equal(tree.children[0].children[2].value, ', unknown [S9.9]');
  assert.equal(tree.children[1].value, '[S1.1]');
  assert.equal(tree.children[2].children[0].type, 'text');
});


test('compact status never conflates generation, saving, memory, or evidence coverage', () => {
  const saved = { turn_state: { generation: 'complete', persistence: 'saved', memory: 'indexed' } };
  assert.equal(getTurnPresentation(saved).label, 'Saved');
  assert.equal(getTurnPresentation(saved).needsAttention, false);
  assert.equal(getTurnPresentation({}).label, 'Status unavailable');
  assert.equal(getTurnPresentation({ ...saved, turn_state: { ...saved.turn_state, memory: 'skipped' } }).label, 'Saved');
  assert.equal(getTurnPresentation({ ...saved, turn_state: { ...saved.turn_state, memory: 'pending' } }).label, 'Saved · memory pending');
  for (const message of [
    { ...saved, unconfirmed: true },
    { ...saved, interruption: 'Connection lost' },
    ...['generation', 'persistence', 'memory'].map((key) => ({ ...saved, turn_state: { ...saved.turn_state, [key]: 'failed' } })),
    { ...saved, turn_state: { ...saved.turn_state, persistence: 'unknown' } },
    { ...saved, metadata: { citation_checks: { items: [{ token: '[S9.9]', status: 'unknown_alias' }] } } },
    { ...saved, metadata: { evidence_snapshot: { sources: [{ status: 'partial' }] } } },
    { ...saved, metadata: { evidence_snapshot: { sources: [{ status: 'success', omitted: ['chunk'] }] } } },
  ]) {
    assert.equal(getTurnPresentation(message).needsAttention, true, JSON.stringify(message));
    assert.equal(getTurnPresentation(message).label, 'Needs attention');
  }
});

test('completed generation awaiting persistence is saving, while interruptions still need attention', () => {
  const pending = { turn_state: { generation: 'complete', persistence: 'pending', memory: 'pending' } };
  for (const message of [pending, { metadata: { generation_state: 'complete', persistence_state: 'pending', memory_state: 'pending' } }]) {
    const view = getTurnPresentation(message);
    assert.equal(view.label, 'Saving…');
    assert.equal(view.needsAttention, false);
    assert.equal(view.settled, false);
  }
  for (const message of [
    { ...pending, unconfirmed: true },
    { ...pending, interruption: 'Connection lost' },
    ...['failed', 'unknown'].map((persistence) => ({ turn_state: { ...pending.turn_state, persistence } })),
    { ...pending, metadata: { evidence_snapshot: { sources: [{ status: 'partial' }] } } },
  ]) {
    assert.equal(getTurnPresentation(message).label, 'Needs attention');
    assert.equal(getTurnPresentation(message).needsAttention, true);
  }
});
