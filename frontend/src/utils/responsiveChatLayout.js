const CHAT_SURFACE_CLASSES = {
  messages: 'mx-auto flex w-full max-w-4xl flex-col gap-4 px-3 py-3 sm:gap-6 sm:px-4 sm:py-4',
  composer: 'mx-auto flex w-full max-w-4xl flex-col gap-2 px-3 py-3 sm:px-4 sm:py-4',
};

export function getChatSurfaceClass(surface = 'messages') {
  return CHAT_SURFACE_CLASSES[surface] || CHAT_SURFACE_CLASSES.messages;
}

export function getTrustRowGridClass() {
  return 'flex flex-wrap items-center gap-1.5';
}

export function getTrustRowCostTileClass() {
  return 'ml-auto flex h-8 items-center gap-2 text-muted-foreground';
}

export function getStageTabListClass() {
  return 'flex gap-2 overflow-x-auto pb-1 sm:flex-wrap';
}
