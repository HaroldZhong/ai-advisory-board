import test from 'node:test';
import assert from 'node:assert/strict';

import {
  RESPONSIVE_MODAL_SIZES,
  getResponsiveModalBodyClass,
  getResponsiveModalContentClass,
} from '../src/utils/responsiveModalLayout.js';

test('responsive modal sizes use viewport-capped widths and heights', () => {
  assert.equal(RESPONSIVE_MODAL_SIZES.wide.widthPx, 960);
  assert.equal(RESPONSIVE_MODAL_SIZES.form.widthPx, 560);
  assert.equal(RESPONSIVE_MODAL_SIZES.narrow.widthPx, 420);
  assert.equal(RESPONSIVE_MODAL_SIZES.narrow.maxHeightVh, 94);

  assert.match(getResponsiveModalContentClass('wide'), /w-\[min\(92vw,960px\)\]/);
  assert.match(getResponsiveModalContentClass('wide'), /max-h-\[min\(90vh,760px\)\]/);
  assert.match(getResponsiveModalContentClass('form'), /w-\[min\(90vw,560px\)\]/);
  assert.match(getResponsiveModalContentClass('narrow'), /w-\[min\(90vw,420px\)\]/);
  assert.match(getResponsiveModalContentClass('narrow'), /max-h-\[min\(94vh,680px\)\]/);
});

test('responsive modal body class keeps header and footer reachable', () => {
  const bodyClass = getResponsiveModalBodyClass();

  assert.match(bodyClass, /min-h-0/);
  assert.match(bodyClass, /overflow-y-auto/);
  assert.match(bodyClass, /overscroll-contain/);
});

test('unknown responsive modal sizes fall back to form sizing', () => {
  assert.equal(
    getResponsiveModalContentClass('unknown'),
    getResponsiveModalContentClass('form'),
  );
});
