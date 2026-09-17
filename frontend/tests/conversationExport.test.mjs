import assert from 'node:assert/strict';
import test from 'node:test';

import { getExportSavedDescription } from '../src/utils/conversationExport.js';

test('getExportSavedDescription shows only the filename', () => {
  assert.equal(
    getExportSavedDescription('C:\\Users\\Test\\AppData\\Local\\HaroldZhong\\AI Advisory Board\\exports\\smoke.md'),
    'Saved smoke.md.',
  );
});

test('getExportSavedDescription handles missing path defensively', () => {
  assert.equal(getExportSavedDescription(''), 'Export saved.');
});

test('export notices handle macOS paths', () => {
  assert.equal(getExportSavedDescription('/Users/test/a b/report.md'), 'Saved report.md.');
});
