export const RESPONSIVE_MODAL_SIZES = {
  wide: {
    widthPx: 960,
    widthVw: 92,
    maxHeightPx: 760,
    maxHeightVh: 90,
    contentClass: 'max-w-none w-[min(92vw,960px)] max-h-[min(90vh,760px)] overflow-hidden',
  },
  analytics: {
    widthPx: 860,
    widthVw: 92,
    maxHeightPx: 720,
    maxHeightVh: 90,
    contentClass: 'max-w-none w-[min(92vw,860px)] max-h-[min(90vh,720px)] overflow-hidden',
  },
  form: {
    widthPx: 560,
    widthVw: 90,
    maxHeightPx: 680,
    maxHeightVh: 88,
    contentClass: 'max-w-none w-[min(90vw,560px)] max-h-[min(88vh,680px)] overflow-hidden',
  },
  narrow: {
    widthPx: 420,
    widthVw: 90,
    maxHeightPx: 680,
    maxHeightVh: 94,
    contentClass: 'max-w-none w-[min(90vw,420px)] max-h-[min(94vh,680px)] overflow-hidden',
  },
};

export function getResponsiveModalContentClass(size = 'form') {
  const config = RESPONSIVE_MODAL_SIZES[size] || RESPONSIVE_MODAL_SIZES.form;
  return config.contentClass;
}

export function getResponsiveModalBodyClass() {
  return 'min-h-0 flex-1 overflow-y-auto overscroll-contain';
}
