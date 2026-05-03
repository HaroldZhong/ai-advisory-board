const CHAT_SURFACE_CLASSES = {
  messages: 'mx-auto flex w-full max-w-4xl flex-col gap-4 px-3 py-3 sm:gap-6 sm:px-4 sm:py-4',
  composer: 'mx-auto flex w-full max-w-4xl flex-col gap-2 px-3 py-3 sm:px-4 sm:py-4',
};

export function getChatSurfaceClass(surface = 'messages') {
  return CHAT_SURFACE_CLASSES[surface] || CHAT_SURFACE_CLASSES.messages;
}

export function getTrustRowGridClass() {
  return 'grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-[1fr_1fr_1.25fr_1fr_1fr_auto]';
}

export function getTrustRowCostTileClass() {
  return 'col-span-1 flex min-h-[52px] items-center justify-between rounded-md border bg-muted/30 px-3 py-2 sm:col-span-2 xl:col-span-1 xl:min-w-[130px] xl:flex-col xl:items-end xl:justify-center';
}

export function getStageTabListClass() {
  return 'flex gap-2 overflow-x-auto pb-1 sm:flex-wrap';
}
