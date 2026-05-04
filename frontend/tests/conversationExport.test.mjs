import assert from 'node:assert/strict';
import test from 'node:test';

import { getExportSavedDescription } from '../src/utils/conversationExport.js';

test('getExportSavedDescription shows the exact saved path', () => {
  assert.equal(
    getExportSavedDescription('C:\\Users\\Test\\AppData\\Local\\HaroldZhong\\AI Advisory Board\\exports\\smoke.md'),
    'Saved to C:\\Users\\Test\\AppData\\Local\\HaroldZhong\\AI Advisory Board\\exports\\smoke.md',
  );
});

test('getExportSavedDescription handles missing path defensively', () => {
  assert.equal(getExportSavedDescription(''), 'Export saved.');
});
